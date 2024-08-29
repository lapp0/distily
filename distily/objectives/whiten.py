# TODO: MODULE CURRENTLY UNUSED

import torch
import torch.nn as nn


class Whitening1dZCA(nn.Module):
    def __init__(self, eps: float = 1e-8):
        """Whitening layer using ZCA Whitening for 1D inputs."""
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computer centered and covariante matrix
        xn = x - x.mean(dim=0, keepdim=True)
        f_cov = torch.mm(xn.t(), xn) / (xn.size(0) - 1)

        # Decompose eigenvalues, regularize, and invert
        fcov_dtype = f_cov.dtype
        eigvals, eigvecs = torch.linalg.eigh(f_cov.float())
        eigvals, eigvecs = eigvals.to(fcov_dtype), eigvecs.to(fcov_dtype)
        inv_sqrt_eigvals = torch.diag(torch.rsqrt(eigvals + self.eps))

        # ZCA whitening matrix
        zca_matrix = eigvecs @ inv_sqrt_eigvals @ eigvecs.t()

        # Apply the ZCA whitening transformation
        return torch.mm(xn, zca_matrix)


class Whitening1dCholesky(nn.Module):
    """
    Whiten features with Cholesky
    References:
    - https://arxiv.org/abs/2403.06213
    - https://github.com/htdt/self-supervised/blob/master/methods/whitening.py
    - https://github.com/roymiles/vkd/issues/2#issuecomment-2182980957
    """
    def __init__(self, eps: float = 1e-8):
        """Whitening layer using Cholesky decomposition for 1D inputs."""
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computer centered and covariante matrix
        xn = x - x.mean(dim=0, keepdim=True)
        f_cov = torch.mm(xn.t(), xn) / (xn.size(0) - 1)

        # Add regularization and compute the inverse square root via Cholesky decomposition
        inv_sqrt = torch.linalg.cholesky(
            (1 - self.eps) * f_cov +
            self.eps * torch.eye(f_cov.size(0), device=x.device)
        )
        inv_sqrt = torch.inverse(inv_sqrt).to(xn.dtype)

        xn = xn.to(inv_sqrt.dtype)

        # Apply the whitening transformation
        return torch.mm(xn, inv_sqrt)


class Whitening1dSVD(nn.Module):
    """Whiten features using Singular Value Decomposition (SVD)."""
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
        cov_dt = cov.dtype
        cov_float64 = cov.float()
        U, S, Vh = torch.linalg.svd(cov_float64, full_matrices=False)
        U = U.to(dtype=cov_dt)
        S = S.to(dtype=cov_dt)

        # Whitening transformation
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(S + self.eps))
        whitened = torch.mm(torch.mm(x_centered, U), D_inv_sqrt)

        return whitened
