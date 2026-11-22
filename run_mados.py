import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic
from torch.utils.data import DataLoader

from gaussian import FIT_ROWS, TokenSpace, quantile_map, radial_correct, sample_indices
from mados import (CLASSES, GRID, IMAGE_SIZE, PATCH, MADOS, VisualFeatures,
                   image_statistics, patch_rows, pixel_features, spectral_tokens)


@dataclass(frozen=True)
class Preset:
    """Validation-selected readout, fusion, and SLIC parameters."""

    bandwidth: tuple | None
    weights: tuple
    segments: int
    beta: float


PRESETS = {
    "cosine": Preset(None, (0.47, 0.29, 0.24), 512, 0.75),
    "kernel": Preset((0.01, 0.10), (0.15, 0.34, 0.51), 256, 0.50),
}


def probabilities(votes, prior, exponent):
    """Divide class votes by their prior raised to a power and normalize."""
    corrected = votes / prior ** exponent
    return corrected / corrected.sum(dim=-1, keepdim=True).clamp_min(1e-30)


class TokenBank:
    """Gaussian token references and their class-fraction labels.

    Parameters
    ----------
    references : torch.Tensor
        Annotated training tokens in Gaussian space.
    labels : torch.Tensor
        Class proportions over the annotated pixels in each reference.
    bandwidth : float or None
        Multiplier of the validation distance scale for the kernel readout.
    exponent : float
        Exponent for class-prior correction.
    """

    def __init__(self, references, labels, bandwidth, exponent):
        self.labels = labels
        self.exponent = exponent
        self.prior = (labels.sum(0) / labels.sum()).clamp_min(1e-6)
        self.unit = F.normalize(references, dim=1).half()
        self.bandwidth = bandwidth
        if bandwidth is not None:
            dtype = torch.float32 if references.shape[1] <= 256 else torch.float16
            self.references = references.to(dtype)
            self.norm2 = self.references.float().square().sum(1)

    def distance2(self, query, calibration=False):
        """Compute squared distances using the archived readout precision."""
        q = query.to(self.references.dtype)
        dot = (q @ self.references.T).float()
        if calibration:
            return (q.float().square().sum(1, keepdim=True) + self.norm2 - 2 * dot).clamp_min_(0)
        return dot.mul_(-2).add_(query.square().sum(1, keepdim=True)).add_(self.norm2).clamp_min_(0)

    def calibrate(self, validation):
        """Set h² from the median squared distance to the closest training token."""
        nearest = torch.cat([self.distance2(q, calibration=True).min(dim=1).values for q in validation.split(512)])
        self.variance = self.bandwidth * nearest.median().item()
        if self.variance <= 0:
            raise ValueError("Gaussian bandwidth is zero. Check the data and checkpoint.")

    def predict(self, query, retrieve=False):
        """Return token probabilities and optionally the pixel branch's patch indices."""
        if self.bandwidth is None or retrieve:
            similarity = F.normalize(query, dim=1).half() @ self.unit.T
        neighbors = similarity.topk(min(500, len(self.labels)), dim=1).indices if retrieve else None
        if self.bandwidth is None:
            sim, indices = similarity.topk(min(50, len(self.labels)), dim=1)
            sim = sim.float()
            weights = ((sim - sim[:, :1]) / 0.1).exp()
            votes = (self.labels[indices] * weights[..., None]).sum(1)
        else:
            distances = self.distance2(query)
            distances -= distances.min(dim=1, keepdim=True).values
            weights = distances.mul_(-1 / (2 * self.variance)).exp_()
            votes = weights @ self.labels
        return probabilities(votes, self.prior, self.exponent), neighbors


class PixelBank:
    """Pixel descriptors and labels in the same reference patch order as the token banks."""

    def __init__(self, train, masks, patch_labels, device):
        # Estimate pixel PCA from all training pixels, including unannotated ones.
        total = torch.zeros(37, dtype=torch.float64, device=device)
        products = torch.zeros((37, 37), dtype=torch.float64, device=device)
        count = 0
        for images, _ in batches(train, 4):
            x = pixel_features(images.to(device)).permute(0, 2, 3, 1).reshape(-1, 37).double()
            total += x.sum(0)
            products += x.T @ x
            count += len(x)
        mean = total / count
        std = (products.diag() / count - mean.square()).clamp_min(1e-12).sqrt()
        corr = (products / count - torch.outer(mean, mean)) / torch.outer(std, std)
        values, basis = torch.linalg.eigh(corr)
        values, basis = values.flip(0), basis.flip(1)
        keep = values >= 1e-3
        self.projection = (basis[:, keep] / (values[keep] + 0.1).sqrt()).float()
        self.mean, self.std = mean.float()[None, :, None, None], std.float()[None, :, None, None]

        # Fit radial correction on pixels of annotated patches, including their unlabeled pixels.
        bank = []
        offset = 0
        for images, _ in batches(train, 4):
            rows = self.whiten(images.to(device))
            mask = masks[offset:offset + len(rows)].to(device)
            bank.append(rows[mask].cpu())
            offset += len(rows)
        white = torch.cat(bank).flatten(0, 1)
        del bank
        sample = white[sample_indices(len(white), FIT_ROWS)]
        sample = sample[sample_indices(len(sample), FIT_ROWS)].to(device)
        self.radial = quantile_map(sample.norm(dim=1, keepdim=True), dimension=white.shape[1])
        self.features = torch.cat([radial_correct(x.to(device), self.radial).half()
                                   for x in white.split(262_144)]).reshape(len(patch_labels), PATCH ** 2, -1)
        self.labels = patch_labels.to(device)

    def whiten(self, images):
        """Whiten pixel features and return them in patch order."""
        x = (pixel_features(images) - self.mean) / self.std
        return patch_rows(torch.einsum("bchw,cd->bdhw", x, self.projection))

    def predict(self, images, neighbors, prior):
        """Return pixel-level probabilities by cosine voting inside each retrieval pool.

        Parameters
        ----------
        images : torch.Tensor
            MADOS reflectances, in batch, band, height, width order.
        neighbors : torch.Tensor
            Visual reference patch indices for each query patch.
        prior : torch.Tensor
            Class prior used when a query has no valid reference votes.

        Returns
        -------
        torch.Tensor
            Pixel probabilities with shape (B, 15, 240, 240).
        """
        white = torch.cat([self.whiten(batch) for batch in images.split(4)])
        query = radial_correct(white.flatten(0, 1), self.radial).half().reshape_as(white)
        votes = []
        chunk = max(1, 800_000_000 // (neighbors.shape[1] * PATCH ** 4))
        for q, nb in zip(query.split(chunk), neighbors.split(chunk)):
            pool = self.features[nb].flatten(1, 2)
            labels = self.labels[nb].flatten(1, 2)
            votes.append(pixel_votes(q, pool, labels, prior))
        p = torch.cat(votes).half().float()
        return (p.reshape(len(images), GRID, GRID, PATCH, PATCH, CLASSES)
                .permute(0, 5, 1, 3, 2, 4).reshape(len(images), CLASSES, IMAGE_SIZE, IMAGE_SIZE))


def pixel_votes(query, pool, labels, prior):
    """Compute the 50-neighbor pixel vote, masking unlabeled reference pixels."""
    similarity = torch.bmm(query, pool.transpose(1, 2))
    norms = query.float().norm(dim=-1, keepdim=True) * pool.float().norm(dim=-1)[:, None, :]
    similarity.div_(norms.clamp_min_(1e-6).half()).masked_fill_(labels[:, None, :] < 0, -torch.inf)
    sim, indices = similarity.topk(min(50, labels.shape[1]), dim=-1)
    sim = sim.float()
    neighbor_labels = labels.gather(1, indices.flatten(1)).reshape_as(indices).long()
    weights = torch.where(sim.isfinite(), ((sim - sim[..., :1]) / 0.05).exp(), 0)
    weighted = F.one_hot(neighbor_labels.clamp_min(0), CLASSES) * weights[..., None]
    counts = weighted.cumsum(dim=-2)[..., -1, :]
    mass = counts.sum(dim=-1, keepdim=True)
    return torch.where(mass > 0, counts / mass.clamp_min(1e-12), prior)


def batches(dataset, batch_size):
    """Read dataset batches in benchmark order."""
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)


def train_tokens(train, encoder, device):
    """Collect training descriptors, the bank mask, and annotated patch labels."""
    visual, spectral, labels = [], [], []
    for images, target in batches(train, 72):
        x = images.to(device)
        visual.append(encoder(x).cpu())
        spectral.append(spectral_tokens(x).cpu())
        labels.append(patch_rows(target[:, None]).squeeze(-1).to(torch.int8))
    labels = torch.cat(labels)
    mask = (labels >= 0).any(dim=1)
    labels = labels[mask]
    fractions = torch.stack([(labels == cls).sum(dim=1) for cls in range(CLASSES)], dim=1).float()
    fractions /= fractions.sum(dim=1, keepdim=True)
    return (torch.cat(visual), torch.cat(spectral)), mask, labels, fractions.to(device)


def calibration_samples(validation, encoder, device):
    """Encode the 4,000 seeded validation tokens used for kernel bandwidths."""
    ids = sample_indices(len(validation) * GRID ** 2, 4000)
    image_ids = ids // (GRID ** 2)
    patches = ids % (GRID ** 2)
    visual = torch.empty((len(ids), 4096), dtype=torch.float16)
    spectral = torch.empty((len(ids), 48))
    start = 0
    for images, _ in batches(validation, 72):
        selection = (image_ids >= start) & (image_ids < start + len(images))
        rows = (image_ids[selection] - start) * GRID ** 2 + patches[selection]
        x = images.to(device)
        visual[selection] = encoder(x)[rows].cpu()
        spectral[selection] = spectral_tokens(x)[rows].cpu()
        start += len(images)
    return visual, spectral


def smooth(prob, guide, preset):
    """Mix the fused scores with their mean in each multispectral superpixel."""
    guide = np.ascontiguousarray(guide.permute(1, 2, 0).cpu().numpy(), dtype=np.float64)
    segments = slic(guide, n_segments=preset.segments, compactness=0.1, channel_axis=-1,
                    start_label=0, convert2lab=False, enforce_connectivity=True)
    _, segment_ids = np.unique(segments.ravel(), return_inverse=True)
    ids = torch.from_numpy(segment_ids).to(prob.device)
    counts = torch.bincount(ids)
    flat = prob.flatten(1)
    mean = flat.new_zeros((CLASSES, len(counts))).index_add_(1, ids, flat) / counts
    return (preset.beta * flat + (1 - preset.beta) * mean[:, ids]).reshape_as(prob)


def add_confusion(confusion, prediction, labels):
    """Accumulate a 15-class confusion matrix over annotated pixels."""
    valid = labels >= 0
    counts = torch.bincount(CLASSES * prediction[valid] + labels[valid], minlength=CLASSES ** 2)
    confusion += counts.reshape(CLASSES, CLASSES)


def mean_iou(confusion):
    """Return PANGAEA's percent mean IoU over all 15 classes."""
    confusion = confusion.double()
    true_positive = confusion.diag()
    return float((true_positive / (confusion.sum(0) + confusion.sum(1) - true_positive + 1e-6)).mean() * 100)


def evaluate(test, encoder, spaces, banks, pixels, guide_stats, preset, device):
    """Run the fixed preset on test scenes and report raw and refined mIoU."""
    raw = torch.zeros((CLASSES, CLASSES), dtype=torch.long, device=device)
    refined = torch.zeros_like(raw)
    mean, std = guide_stats
    weights = torch.tensor(preset.weights, device=device)
    for i, (images, labels) in enumerate(batches(test, 72)):
        images, labels = images.to(device), labels.to(device)
        visual = encoder(images).split(9 * GRID ** 2)
        spectral = spectral_tokens(images).split(9 * GRID ** 2)
        for x, y, v, s in zip(images.split(9), labels.split(9), visual, spectral):
            vp, neighbors = banks[0].predict(spaces[0](v), retrieve=True)
            sp, _ = banks[1].predict(spaces[1](s))
            token_maps = [F.interpolate(p.half().float().reshape(-1, GRID, GRID, CLASSES).permute(0, 3, 1, 2),
                                        size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
                          for p in (vp, sp)]
            pixel_maps = pixels.predict(x, neighbors, banks[0].prior)
            log_maps = torch.stack([p.clamp_min(1e-6).log() for p in (*token_maps, pixel_maps)])
            scores = torch.einsum("g,gbchw->bchw", weights, log_maps).softmax(dim=1)
            add_confusion(raw, scores.argmax(1), y)
            for prob, guide, target in zip(scores, (x - mean) / std, y):
                add_confusion(refined, smooth(prob, guide, preset).argmax(0), target)
        print(f"Evaluated {min(72 * (i + 1), len(test))}/{len(test)} images", flush=True)
    return mean_iou(raw), mean_iou(refined)


@torch.inference_mode()
def main():
    """Fit the statistics and evaluate one full-label MADOS preset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted MADOS directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="timm DINOv3 ViT-L/16 safetensors file")
    parser.add_argument("--predictor", choices=PRESETS, default="cosine", help="Token readout (default: cosine)")
    args = parser.parse_args()
    if not torch.cuda.is_bf16_supported():
        parser.error("The full MADOS reproduction requires CUDA with bfloat16 support.")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    preset = PRESETS[args.predictor]
    train, val, test = (MADOS(args.data_root, split) for split in ("train", "val", "test"))
    print(f"MADOS split sizes: {len(train)} train, {len(val)} val, {len(test)} test", flush=True)
    stretches, mean, std = image_statistics(train, device)
    encoder = VisualFeatures(args.checkpoint, stretches, device)
    print("Encoding training tokens", flush=True)
    features, mask, patch_labels, fractions = train_tokens(train, encoder, device)
    print(f"Fitting token spaces on training features ({int(mask.sum())} annotated reference patches)", flush=True)
    spaces = [TokenSpace.fit(x, d, device) for x, d in zip(features, (768, 48))]
    bandwidths = preset.bandwidth or (None, None)
    banks = [TokenBank(torch.cat([space(rows).detach() for rows in x[mask].split(65_536)]), fractions, bandwidth, exponent)
             for space, x, bandwidth, exponent in zip(spaces, features, bandwidths, (0.0, 0.5))]
    del features
    if preset.bandwidth:
        print("Calibrating Gaussian bandwidths on validation tokens", flush=True)
        samples = calibration_samples(val, encoder, device)
        for bank, space, sample in zip(banks, spaces, samples):
            bank.calibrate(space(sample))
        print(f"Squared kernel bandwidths: {banks[0].variance:.4f}, {banks[1].variance:.4f}", flush=True)
    print("Fitting pixel space and building its reference bank", flush=True)
    pixels = PixelBank(train, mask, patch_labels, device)
    raw, refined = evaluate(test, encoder, spaces, banks, pixels, (mean, std), preset, device)
    print(f"{args.predictor} test mIoU: {refined:.2f}% ({raw:.2f}% before SLIC)")


if __name__ == "__main__":
    main()
