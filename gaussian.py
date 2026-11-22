from dataclasses import dataclass

import torch
from scipy.stats import chi2

FIT_ROWS = 200_000


def sample_indices(length, limit):
    """Sample row indices with the seed used in the paper runs."""
    return torch.randperm(length, generator=torch.Generator().manual_seed(0))[:limit]


@dataclass
class TokenSpace:
    """Fitted PCA, marginal map, re-whitening, and radial map for a token group."""

    mean: torch.Tensor
    std: torch.Tensor
    basis: torch.Tensor
    scale: torch.Tensor
    marginal: tuple
    center: torch.Tensor
    rotation: torch.Tensor
    radial: tuple

    @classmethod
    def fit(cls, tokens, dimension, device):
        """Fit the token transform on up to 200,000 training descriptors.

        Parameters
        ----------
        tokens : torch.Tensor
            Training token rows, including tokens without annotated pixels.
        dimension : int
            Number of principal components retained by initial whitening.
        device : torch.device
            Device used for the covariance, eigensolver, and quantile maps.

        Returns
        -------
        TokenSpace
            Transform to apply unchanged to the reference and query rows.
        """
        sample = tokens[sample_indices(len(tokens), FIT_ROWS)].to(device, torch.float32)
        mean, std = sample.mean(0), sample.std(0).clamp_min(1e-6)
        standardized = (sample - mean) / std
        values, basis = torch.linalg.eigh(standardized.T @ standardized / (len(sample) - 1))
        basis = basis.flip(1).contiguous()[:, :dimension]
        scale = values.flip(0)[:dimension].clamp_min(1e-8).sqrt()
        del sample, standardized

        # Compute in the same row blocks as the reference PCA projection.
        whitened = torch.cat([(((rows.to(device, torch.float32) - mean) / std) @ basis / scale).cpu()
                              for rows in tokens.split(100_000)])
        sample_ids = sample_indices(len(whitened), FIT_ROWS)
        sample = whitened[sample_ids].to(device)
        marginal = quantile_map(sample)
        mapped = interpolate(sample, marginal)
        center = mapped.mean(0)
        centered = mapped - center
        values, vectors = torch.linalg.eigh(centered.T @ centered / (len(centered) - 1))
        rotation = vectors / values.clamp_min(1e-6).sqrt()
        del sample, mapped, centered
        # Match the reference blocks before sampling the re-whitened norms.
        norms = []
        for rows in whitened.split(65_536):
            rewhitened = (interpolate(rows.to(device), marginal) - center) @ rotation
            norms.append(rewhitened.norm(dim=1, keepdim=True).cpu())
        radial = quantile_map(torch.cat(norms)[sample_ids].to(device), dimension=rotation.shape[1])
        return cls(mean, std, basis, scale, marginal, center, rotation, radial)

    def __call__(self, tokens):
        """Apply the fitted token transform to a block of descriptor rows."""
        z = ((tokens.to(self.mean.device, torch.float32) - self.mean) / self.std) @ self.basis / self.scale
        z = (interpolate(z, self.marginal) - self.center) @ self.rotation
        return radial_correct(z, self.radial)


def quantile_map(sample, dimension=None):
    """Fit a 1,025-knot marginal normal map or a radial chi map.

    Parameters
    ----------
    sample : torch.Tensor
        Sample rows by feature. A radial sample has a single norm column.
    dimension : int, optional
        Degrees of freedom of the target chi distribution for a radial map.

    Returns
    -------
    tuple of torch.Tensor
        Empirical knots by feature and the shared target knots.
    """
    n = len(sample)
    normal = torch.distributions.Normal(0.0, 1.0)
    bound = -normal.icdf(torch.tensor(1.0 / (n + 1))).item()
    target = torch.linspace(-bound, bound, 1025, device=sample.device)
    positions = (normal.cdf(target) * (n - 1)).round().long().clamp(0, n - 1)
    knots = torch.cat([cols.sort(dim=0).values[positions].T for cols in sample.split(256, dim=1)]).contiguous()
    span = (knots[:, -1:] - knots[:, :1]).clamp_min(1e-6)
    knots += torch.arange(len(target), device=sample.device) * span * 1e-7
    if dimension is not None:
        probabilities = normal.cdf(target.cpu()).double().numpy()
        target = torch.from_numpy(chi2.ppf(probabilities, dimension) ** 0.5).to(sample)
    return knots, target


def interpolate(values, mapping):
    """Apply a piecewise linear empirical map with endpoint clamping."""
    knots, target = mapping
    x = values.T.contiguous()
    i = torch.searchsorted(knots, x).clamp(1, len(target) - 1)
    lower, upper = knots.gather(1, i - 1), knots.gather(1, i)
    t = ((x - lower) / (upper - lower).clamp_min(torch.finfo(x.dtype).tiny)).clamp(0, 1)
    return (target[i - 1] + t * (target[i] - target[i - 1])).T


def radial_correct(values, mapping):
    """Map the vector norms while preserving their directions."""
    norms = values.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return values * interpolate(norms, mapping) / norms
