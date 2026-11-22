import warnings
from glob import glob
from pathlib import Path

import numpy as np
import rasterio
import timm
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from torch.utils.data import Dataset

IMAGE_SIZE = 240
PATCH = 16
GRID = IMAGE_SIZE // PATCH
CLASSES = 15
VIEWS = ((3, 2, 1), (7, 3, 2), (6, 5, 4), (10, 9, 8))
SPLIT_SIZES = {"train": 1433, "val": 642, "test": 728}


class MADOS(Dataset):
    """MADOS crops in PANGAEA's scene and crop order.

    Parameters
    ----------
    root : pathlib.Path
        Extracted dataset directory containing scenes and ``splits/``.
    split : {'train', 'val', 'test'}
        Benchmark split to load.
    """

    def __init__(self, root, split):
        split_file = root / "splits" / f"{split}_X.txt"
        split_ids = set(split_file.read_text().split())
        self.samples = []
        found = set()
        for scene in sorted(root.glob("Scene_*")):
            # PANGAEA sorts scenes and uses the on-disk mask order within each scene.
            for mask in glob(str(scene / "10" / "*_cl_*.tif")):
                crop = mask.rsplit("_cl_", 1)[1]
                crop_id = f"{scene.name}_{Path(crop).stem}"
                if crop_id not in split_ids:
                    continue
                paths = sorted(scene.glob(f"*/*L2R_rhorc*_{crop}"), key=self.band_number)
                if len(paths) != 11:
                    raise ValueError(f"Expected 11 reflectance bands for {crop_id}, found {len(paths)}")
                self.samples.append((paths, mask))
                found.add(crop_id)
        if found != split_ids or len(self.samples) != SPLIT_SIZES[split]:
            raise ValueError(f"{split_file}: found {len(self.samples)} of {SPLIT_SIZES[split]} expected crops")

    @staticmethod
    def band_number(path):
        """Return the numeric band field in a MADOS filename."""
        return int(path.name.split("_")[-2])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Return the reflectance tensor and zero-based labels, with -1 ignored."""
        paths, mask = self.samples[index]
        bands = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            for band in paths:
                with rasterio.open(band) as tif:
                    bands.append(tif.read(1, out_shape=(IMAGE_SIZE, IMAGE_SIZE), resampling=Resampling.nearest))
            with rasterio.open(mask) as tif:
                labels = tif.read(1).astype(np.int64) - 1
        image = np.stack(bands).astype(np.float32, copy=False)
        image[np.isnan(image)] = 0
        return torch.from_numpy(image), torch.from_numpy(labels)


def image_statistics(train, device):
    """Fit the four view stretches and the standardization used by SLIC.

    Parameters
    ----------
    train : MADOS
        Full training split, in benchmark order.
    device : torch.device
        Device for the returned statistics.

    Returns
    -------
    stretches : list of tuple of torch.Tensor
        Lower and upper reflectance bounds of each three-channel view.
    mean, std : torch.Tensor
        Channel statistics of shape (11, 1, 1) for the SLIC guide.
    """
    sample = np.random.default_rng(0).choice(len(train), 200, replace=False)
    s = torch.stack([train[int(i)][0] for i in sample])
    low, high = {}, {}
    for band in sorted(set().union(*VIEWS)):
        values = s[:, band].reshape(-1)
        values = values[values != 0]
        order = torch.randperm(len(values), generator=torch.Generator().manual_seed(0))[:2_000_000]
        low[band], high[band] = torch.quantile(values[order], torch.tensor([0.01, 0.99]))
    stretches = [tuple(torch.stack([bound[b] for b in view]).to(device)[None, :, None, None]
                       for bound in (low, high)) for view in VIEWS]
    sample = np.random.default_rng(0).choice(len(train), 100, replace=False)
    s = torch.stack([train[int(i)][0] for i in sample])
    return stretches, s.mean((0, 2, 3)).to(device)[:, None, None], s.std((0, 2, 3)).clamp_min(1e-6).to(device)[:, None, None]


class VisualFeatures:
    """Four-view frozen DINOv3 ViT-L/16 patch encoder.

    Parameters
    ----------
    checkpoint : pathlib.Path
        timm-format DINOv3 LVD-1689M safetensors checkpoint.
    stretches : list of tuple of torch.Tensor
        One fitted low/high pair per view, in ``VIEWS`` order.
    device : torch.device
        Device for the frozen encoder.
    """

    def __init__(self, checkpoint, stretches, device):
        self.model = timm.create_model("vit_large_patch16_dinov3.lvd1689m", num_classes=0,
                                       pretrained=False, checkpoint_path=str(checkpoint))
        self.model.requires_grad_(False).to(device).eval()
        self.stretches = stretches
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=device)[None, :, None, None]
        self.std = torch.tensor(cfg["std"], device=device)[None, :, None, None]

    def __call__(self, images):
        """Return flattened patch tokens with the four 1,024-D views concatenated."""
        features = []
        for bands, (low, high) in zip(VIEWS, self.stretches):
            view = ((images[:, bands] - low) / (high - low)).clamp(0, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=images.is_cuda):
                tokens = self.model.forward_features((view - self.mean) / self.std)
            tokens = F.normalize(tokens[:, self.model.num_prefix_tokens:].float(), dim=-1)
            features.append(tokens.flatten(0, 1).half())
        return torch.cat(features, dim=-1)


def spectral_indices(images):
    """Return NDVI, NDWI, NDBI, NBR, and BSI maps in that order."""
    blue, green, red, nir, swir1, swir2 = (images[:, i] for i in (1, 2, 3, 7, 9, 10))

    def difference(a, b):
        return (a - b) / (a + b + 1e-6)

    return torch.stack((difference(nir, red), difference(green, nir), difference(swir1, nir),
                        difference(nir, swir2), difference(swir1 + red, nir + blue)), dim=1)


def patch_rows(maps):
    """Convert (B, C, H, W) maps to (B × 225, 256, C) in patch order."""
    return (maps.reshape(-1, maps.shape[1], GRID, PATCH, GRID, PATCH)
            .permute(0, 2, 4, 3, 5, 1).reshape(-1, PATCH * PATCH, maps.shape[1]))


def spectral_tokens(images):
    """Return per-patch mean, population std, and mean gradient for 16 channels."""
    maps = torch.cat((images, spectral_indices(images)), dim=1)
    patches = (maps.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
               .permute(0, 2, 3, 1, 4, 5).reshape(-1, 16, PATCH, PATCH))
    flat = patches.flatten(-2)
    corner = patches[..., :-1, :-1]
    dx = patches[..., :-1, 1:] - corner
    dy = patches[..., 1:, :-1] - corner
    gradient = (dx.square() + dy.square()).sqrt().mean((-2, -1))
    features = torch.stack((flat.mean(-1), flat.std(-1, correction=0), gradient), dim=-1)
    return torch.nan_to_num(features.flatten(1), nan=0, posinf=0, neginf=0)


def pixel_features(images):
    """Return pixel bands, four indices, and reflected 3 × 3 band statistics."""
    kernel = images.new_full((11, 1, 3, 3), 1 / 9)
    padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
    mean = F.conv2d(padded, kernel, groups=11)
    std = (F.conv2d(padded.square(), kernel, groups=11) - mean.square()).clamp_min(0).sqrt()
    return torch.nan_to_num(torch.cat((images, spectral_indices(images)[:, :4], mean, std), dim=1),
                            nan=0, posinf=0, neginf=0)
