import torch
import torch.nn as nn


class OLD_Whitening1d(nn.Module):
    """
    Whiten features.

    References:
    - https://arxiv.org/abs/2403.06213
    - https://github.com/htdt/self-supervised/blob/master/methods/whitening.py
    - https://github.com/roymiles/vkd/issues/2#issuecomment-2182980957
    """
    def __init__(self, features: torch.Tensor, eps: float = 1e-8):
        """Whitening layer using Cholesky decomposition for 1D inputs."""
        super().__init__()
        self.feature_size = features.size(-1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computer centered and covariante matrix
        xn = x - x.mean(dim=0, keepdim=True)
        f_cov = torch.mm(xn.t(), xn) / (xn.size(0) - 1)

        # Add regularization and compute the inverse square root via Cholesky decomposition
        inv_sqrt = torch.linalg.cholesky(
            (1 - self.eps) * f_cov +
            self.eps * torch.eye(self.feature_size, device=x.device)
        )
        inv_sqrt = torch.inverse(inv_sqrt).to(xn.dtype)

        xn = xn.to(inv_sqrt.dtype)
        # Apply the whitening transformation
        return torch.mm(xn, inv_sqrt)


class Whitening1d(nn.Module):
    """Whiten features"""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # Centering
        x_centered = x - x.mean(dim=0, keepdim=True)

        # Covariance matrix
        cov = torch.mm(x_centered.T, x_centered) / (x_centered.size(0) - 1)

        # Ensure symmetry and add regularization
        cov = (cov + cov.T) / 2
        cov += torch.eye(cov.size(0), device=x.device) * self.eps

        # Singular Value Decomposition (SVD)
        U, S, Vh = torch.linalg.svd(cov, full_matrices=False)

        # Whitening transformation
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(S + self.eps))
        whitened = torch.mm(torch.mm(x_centered, U), D_inv_sqrt)

        return whitened
