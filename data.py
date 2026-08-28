from contextlib import contextmanager
from pathlib import Path
import random
import datasets
from ml_collections import ConfigDict
import numpy as np
import tifffile
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from skimage.measure import regionprops
from ucell.utils import pad_channel


@contextmanager
def rank_zero_first():
    '''Run the body on rank 0 before any other rank enters it.

    datasets' .map()/.filter() write a cache file next to the dataset. If every
    rank builds a cold cache at once they race, and a rank that mmaps a region
    another rank is still writing dies with SIGBUS. Letting rank 0 build the
    cache first means the other ranks only ever read a finished file.

    A no-op when torch.distributed is not initialized, so single-process
    callers are unaffected.
    '''
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    if distributed and rank != 0:
        dist.barrier()  # wait for rank 0 to finish building the cache

    try:
        yield
    finally:
        # in a finally so a failure on rank 0 releases the others rather than
        # hanging them at the barrier forever
        if distributed and rank == 0:
            dist.barrier()

def random_flip(image, label):
    if random.random() >= .5:
        image = image[::-1]
        label = label[::-1] * (-1, 1, 1)
    if random.random() >= .5:
        image = image[:, ::-1]
        label = label[:, ::-1] * (1, -1, 1)
    if random.random() >= .5:
        image = image.transpose(1,0,2)
        label = label[..., (1,0,2)].transpose(1,0,2)

    return image, label
    

def format_and_augment(example, *, imagesize = 256, augment=True):
    image = example['image'].astype("float32")
    label = example['label'].astype("float32")
    sz = example['sz']

    if label.shape[0] == 2:
        segmentation = (label != 0).any(axis=0, keepdims=True).astype(label.dtype)
        label = np.r_[label, segmentation]

    assert label.shape[0] == 3

    # preprocess
    image = pad_channel(image)
    image = image / (image.max() + 1e-5)

    label = np.moveaxis(label, 0, -1)

    if augment:
        image, label = random_flip(image, label)

    # combine
    combined = np.c_[image, label]

    if not augment:
        T = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(size=imagesize, pad_if_needed=True),
            # v2.CenterCrop(imagesize),
        ])
        combined = T(combined)

    else:
        scaling = 2 ** (0.6 * random.normalvariate(0, 1)) * (sz / 35)
        aniso = 2 ** (0.2 * random.normalvariate(0, 1))
        scaling = np.clip([scaling * aniso, scaling / aniso], 0.3, 3.3)
        height, width = (imagesize * scaling).astype(int)

        T = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(size=(height, width), pad_if_needed=True),
            v2.Resize(size=(imagesize, imagesize), antialias=False),
        ])
        combined = T(combined)

    image = combined[:3]
    label = combined[3:]
    task_id = example['task_id'] if 'task_id' in example else 0

    return dict(image=image, label=label, task_id=task_id)


def get_sz(fn, default_sz=35.):
    mask = tifffile.imread(fn)
    props = regionprops(np.maximum(0, mask.astype(int)))
    if len(props) == 0:
        return default_sz
    else:
        return sum([p.axis_major_length for p in props]) / len(props)


def compute_label(example):
    from ucell.dynamics import masks_to_flows_gpu
    from ucell.utils import clean_up_mask
    from PIL import Image

    if not "label" in example:
        if "flow" in example:
            flow_y, flow_x = example['flow']
            segmentation = (np.array(flow_y).astype("float32") != 0) | (np.array(flow_x).astype("float32") != 0)
    
            example['label'] = [flow_y, flow_x, Image.fromarray(segmentation.astype("float32"))]
    
        else:
            mask = np.array(example['masks'])
            segmentation = (mask != 0).astype("float32")
            mask = clean_up_mask(np.maximum(mask, 0))
            flow = np.asarray(masks_to_flows_gpu(mask, device=torch.device("cuda"))).astype("float32")

            example['label'] = [Image.fromarray(flow[i]) for i in range(2)] + [Image.fromarray(segmentation)]

    return example


def load_img_folder(config, split):
    '''
    Create a HF dataset from config.data_dir. Support three types of dataset
    format:

    1. A preprocessed and saved HF dataset will be loaded without change. Useful
       for fast processing of very large datasets
    2. A folder with image files and a metadata.csv file. Useful for overriding
       aux features such as "sz" and "task_id"
    3. A folder with just paired image and mask files, where mask files are named
       with "_label" suffix. Image features such as "sz" will be regenerated on
       the fly, and "task_id" will be set to config.task_id.     
    '''

    ds = None
    
    p = Path(config.data_dir) / split
    if p.exists():
        # rank 0 builds the map/filter cache first; the other ranks then read
        # the finished file instead of racing to write it
        with rank_zero_first():
            if (p/"dataset_info.json").exists():
                ds = (
                    datasets.load_from_disk(p)
                    .map(compute_label, writer_batch_size=16, num_proc=0)
                )

            elif (p/"metadata.csv").exists():
                ds = (
                    datasets.load_dataset(str(p), split="train")
                    .map(compute_label, writer_batch_size=16, num_proc=0)
                    .remove_columns("masks")
                )

            else:
                # Sorted so every rank builds the same dataset, and therefore
                # the same cache fingerprint, as rank 0
                mask_fns = sorted(p.glob("*_label.tif"))
                img_fns = [fn.with_name(fn.name.replace("_label", "")) for fn in mask_fns]
                szs = [get_sz(fn) for fn in mask_fns]

                ds = (
                    datasets.Dataset.from_dict({
                        "image": [str(fn) for fn in img_fns],
                        "masks": [str(fn) for fn in mask_fns],
                        "sz": szs,
                        "task_id": [config.task_id] * len(img_fns),
                    })
                    .cast_column("image", datasets.features.Image())
                    .cast_column("masks", datasets.features.Image())
                    .filter(lambda x: np.array(x['masks']).max() > 0)
                    .map(compute_label, writer_batch_size=16, num_proc=0, load_from_cache_file=True)
                    .remove_columns("masks")
                )

        if config.task_id == -1:
            ds = ds.remove_columns("task_id")

    return ds


class WeightedDistributedSampler(torch.utils.data.Sampler):
    """Sample images with per-task weights, sharding across ranks itself.

    The training set is a concatenation of subsets of very unequal size, so
    uniform shuffling shows each subset in proportion to its image count --
    livecell 41% of batches against monusac's 2.7%.  Weighting each image by
    `n_images_in_its_task ** -alpha` corrects that: alpha=0 leaves the natural
    proportions, alpha=1 gives every task an equal share, and values between
    interpolate.

    Full balancing is rarely what you want here.  Equalising a 209-image subset
    against a 3189-image one means repeating each of those 209 roughly 15 times
    per epoch, which overfits them long before the large subsets converge.

    This sampler shards by rank on its own rather than being wrapped in a
    DistributedSampler, so callers must pass use_distributed_sampler=False to
    fabric.setup_dataloaders.  All ranks draw from one generator seeded
    identically and then take disjoint strides of the result, so the global
    draw is consistent and no index is handed to two ranks.
    """

    def __init__(self, weights, num_samples, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if distributed else 1
        self.rank = dist.get_rank() if distributed else 0
        # floor, so every rank yields the same count and no rank runs short
        self.num_samples = num_samples // self.world_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        # bumped per pass so successive epochs differ; every rank calls
        # __iter__ the same number of times, so the seeds stay in step
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        idx = torch.multinomial(
            self.weights, self.num_samples * self.world_size,
            replacement=True, generator=g,
        )
        yield from idx[self.rank::self.world_size].tolist()


def task_weights(task_ids, alpha):
    """Per-image sampling weight, flattening the task histogram by alpha."""
    task_ids = np.asarray(task_ids)
    tasks, counts = np.unique(task_ids, return_counts=True)
    per_task = counts.astype("float64") ** (-alpha)
    return per_task[np.searchsorted(tasks, task_ids)]


def get_dataloader(config:"ConfigDict", split:str)->DataLoader|None:
    collapse = config.get("collapse_task_id", -1)

    def collate_fn(examples):
        augmented = []
        for example in examples:
            out = format_and_augment(
                example,
                augment=split=="train", 
                imagesize=config.image_size,
            )
            if collapse >= 0:
                # the dataset's own task_id still drives the sampler weights;
                # only what the model is told is overridden
                out["task_id"] = collapse
            augmented.append(out)
        return torch.utils.data.default_collate(augmented)

    ds = load_img_folder(config, split)
    if ds is None:
        return None

    alpha = config.get("sampling_alpha", 0.0)
    weights = None
    if split == "train" and alpha > 0 and "task_id" in ds.column_names:
        # read before .repeat() so only one copy is materialized, then tile
        weights = task_weights(ds.select_columns(["task_id"])["task_id"], alpha)

    repeats = config.epochs_per_iter if split == "train" else 1
    ds = ds.with_format('numpy').repeat(repeats)

    sampler = None
    if weights is not None:
        sampler = WeightedDistributedSampler(
            np.tile(weights, repeats), len(ds), seed=config.seed,
        )

    dataloader = DataLoader(
        ds, 
        batch_size=config.batch_size,
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=config.dataloader_workers,
        drop_last=True,
        pin_memory=True,
    )

    return dataloader
