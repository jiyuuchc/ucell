from functools import partial, wraps
import numpy as np

def remove_small_instances(mask, *, min_area=0, cleanup=False):
    """ remove all instances smaller than min_area"""
    if min_area <= 0:
        return mask
    
    assert mask.min() >= 0, f"Mask has negative values."

    unique_ids, counts = np.unique(mask, return_counts=True)

    lut = np.zeros([unique_ids[-1] + 1], dtype=mask.dtype)
    non_small_ids = unique_ids[counts >= min_area]

    if cleanup:
        if non_small_ids[0] == 0:
            non_small_ids = non_small_ids[1:]
        lut[non_small_ids] = np.arange(len(non_small_ids)) + 1
    else:
        lut[non_small_ids] = non_small_ids

    return lut[mask]


def clean_up_mask(mask):
    """ ensure continuity of mask ID"""
    unique_ids = np.unique(mask)

    assert np.all(unique_ids >= 0), f"Mask containes negative values."

    lut = np.zeros(unique_ids.max()+1, dtype=mask.dtype)
    lut[unique_ids] = np.arange(len(unique_ids))

    return lut[mask]


def remove_border_instances(mask):
    """ remove all instances at image border"""
    if mask.ndim == 2:
        border_pixels = set(
            mask[0, :].tolist()
            + mask[:, -1].tolist()
            + mask[-1, :].tolist()
            + mask[:, 0].tolist()
        )
    else:
        assert mask.ndim == 3, "mask must be 2D or 3D"
        border_pixels = set(
            mask[0, :, :].reshape(-1).tolist()
            + mask[-1, :, :].reshape(-1).tolist()
            + mask[:, 0, :].reshape(-1).tolist()
            + mask[:, -1, :].reshape(-1).tolist()
            + mask[:, :, 0].reshape(-1).tolist()
            + mask[:, :, -1].reshape(-1).tolist()
        )

    if 0 in border_pixels:
        border_pixels.remove(0)
    
    mask = np.where(
        np.isin(mask, list(border_pixels)),
        0,
        mask,
    )
    
    return mask


def pad_channel(image):
    if image.ndim == 2:
        image = image[..., None]

    C = image.shape[-1]
    if C == 1:
        image = np.c_[image, image, image]
    if C == 2:
        image = np.c_[image, np.zeros_like(image[..., :1])]

    assert image.shape[-1] == 3

    return image


def center_crop(image, mask=None, *, crop_size=512):
    """ Take a (image, mask) pair. Return their center crop
    (or padding if needed) of the size (crop_size, crop_size).
    """
    if image.ndim == 2:
        image = image[...,None]

    H, W, C = image.shape
    assert mask is None or mask.shape == (H, W)

    if H < crop_size or W < crop_size:
        padding = [
            [0, max(0, crop_size-H)],
            [0, max(0, crop_size-W)],
        ]
        image = np.pad(image, padding + [[0,0]])

        if mask is not None:
            mask = np.pad(mask, padding)
    
        H, W, C = image.shape

    image = image[
        (H-crop_size)//2:(H+crop_size)//2,
        (W-crop_size)//2:(W+crop_size)//2,
        :
    ]

    if mask is not None:
        mask = mask[
            (H-crop_size)//2:(H+crop_size)//2,
            (W-crop_size)//2:(W+crop_size)//2,
        ]

        return image, mask

    else:
        return image



def to_patches(images, *, GS=256):
    def _to_patches(image):
        H, W, _ = image.shape
        padding = [[0, max(0, GS-H)], [0, max(0, GS-W)], [0,0]]
        image = np.pad(image, padding)

        overlap = GS // 8
        H, W, _ = image.shape
        stack = []
        for yc in range(0, H - overlap, GS-overlap):
            for xc in range(0, W - overlap, GS-overlap):
                if yc + GS > H: yc = H - GS
                if xc + GS > W: xc = W - GS
                patch = image[yc:yc+GS, xc:xc+GS]
                assert patch.shape[:2] == (GS, GS)
                stack.append(patch)

        return np.stack(stack)

    if images.ndim == 3:
        images = images[None]
    
    assert images.ndim == 4, f"invalid input image dim {images.shape}"

    return np.concatenate([_to_patches(x) for x in images])


def from_patches(patches, orig_shape, *, GS=256):
    squeeze = len(orig_shape) == 3
    if squeeze:
        orig_shape = (1,) + tuple(orig_shape)
    
    B, H0, W0, C = orig_shape
    
    overlap = GS // 8
    
    def _from_patches(section):
        H, W = max(H0, GS), max(W0, GS)
        y = np.zeros((H, W, C))
        cnts = np.zeros((H, W, 1))

        k = 0
        for yc in range(0, H - overlap, GS-overlap):
            for xc in range(0, W - overlap, GS-overlap):
                if yc + GS > H: yc = H - GS
                if xc + GS > W: xc = W - GS

                cnts[yc:yc+GS, xc:xc+GS] += 1
                y[yc:yc+GS, xc:xc+GS] += section[k]
                k += 1

        y = (y / cnts)[:H0, :W0, :]

        return y

    patches = patches.reshape(B, -1, * patches.shape[1:])

    y = np.stack([_from_patches(p) for p in patches])

    assert y.shape == orig_shape

    if squeeze: y = y.squeeze(0)

    return y


def patcherize(fn=None, *, GS=256, B=16):
    """ wraps a image processing fuction whose input needs be a 4d tensor of specific dim, ie. image patchs.
        The functon should return a ndarray of the same shape as input
    Args:
        fn: f: (B, GS, GS, C) -> (B, GS, GS, C)
        GS: patch size
        B: batch size
    """
    if fn is None:
        return partial(patcherize, GS=GS, B=B)

    @wraps(fn)
    def _f(x, *args, **kwargs):
        orig_shape = x.shape

        x_patches = to_patches(x, GS=GS)

        n_patches = x_patches.shape[0]
        pad_to = (n_patches - 1) // B * B + B
        x_patches = np.pad(x_patches, [[0, pad_to - n_patches], [0, 0], [0, 0], [0, 0]])

        y_patches = []
        for k in range(0, pad_to, B):
            y_patches.append(np.asarray(fn(x_patches[k:k+B], *args, **kwargs)))

        y_patches = np.concatenate(y_patches)

        y = from_patches(y_patches, orig_shape, GS=GS)

        return y

    return _f

def show_images(imgs, locs=None, **kwargs):
    import matplotlib.patches
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 5))
    if len(imgs) == 1:
        axs = [axs]

    for k, img in enumerate(imgs):
        axs[k].imshow(img, **kwargs)
        axs[k].axis("off")
        if locs is not None and locs[k] is not None:
            loc = np.round(locs[k]).astype(int)
            for p in loc:
                c = matplotlib.patches.Circle(
                    (p[1], p[0]), fill=False, edgecolor="white"
                )
                axs[k].add_patch(c)
    plt.tight_layout()