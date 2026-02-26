from functools import partial
import random
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

task_id_map = {
    'cellpose': 0,
    "tissuenet": 1,
    "livecell": 2,
    'nips': 3,
}

def pad_channel(image):
    if image.ndim == 2:
        image = image[...,None]

    if image.shape[0] <= 3:
        image = np.moveaxis(image, 0, -1)

    if image.shape[-1] == 1:
        image = np.tile(image, (1, 1, 3))
    elif image.shape[-1] == 2:
        image = np.pad(image, [[0, 0], [0, 0], [0, 1]])
    return image


def random_flip(image, label):
    if random.random() >= .5:
        image = image[::-1]
        label = label[::-1] * (-1, 1)
    if random.random() >= .5:
        image = image[:, ::-1]
        label = label[:, ::-1] * (1, -1)
    if random.random() >= .5:
        image = image.transpose(1, 0, 2)
        label = label[..., (1,0)].transpose(1, 0, 2)

    return image, label
    

def format_and_augment(example, *, imagesize = 256, augment=True):
    image, label, sz = example['image'], example['flow'], example['sz']
    if example['src'].startswith("omni"):
        sz /= 1.6

    # preprocess
    image = pad_channel(image)
    label = np.moveaxis(label, 0, -1)

    if augment:
        image, label = random_flip(image, label)

    # combine
    label_mask = (label != 0).any(axis=-1, keepdims=True).astype(image.dtype)
    combined = np.c_[image, label, label_mask]

    if not augment:
        T = v2.Compose([
            v2.ToImage(),
            v2.CenterCrop(imagesize),
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

    image = combined[:3] / 256
    label = combined[3:]
    task_id = task_id_map[example['src']]

    return dict(image=image, label=label, task_id=task_id)


def scs(config, split):
    def collate_fn(examples):
        augmented = []
        for example in examples:
            augmented.append(format_and_augment(
                example,
                augment=split=="train", 
                imagesize=config.image_size)
            )
        
        return torch.utils.data.default_collate(augmented)

    ds = (
        datasets.load_dataset(
            "jiyuuchc/scs", 
            split=split,
            token=config.token if len(config.token) > 0 else None,
        )
        .with_format('numpy')
        .repeat(config.epochs_per_iter if split == 'train' else 1)
    )

    dataloader = DataLoader(
        ds, 
        batch_size=config.batch_size,
        shuffle=split=="train",
        collate_fn=collate_fn,
        num_workers=config.dataloader_workers,
        drop_last=True,
        pin_memory=True,
    )

    return dataloader
