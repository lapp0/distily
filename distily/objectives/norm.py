import torch
import torch.nn as nn


class Whitening1d(nn.Module):
    """
    Whiten features.

    References:
    - https://arxiv.org/abs/2403.06213
    - https://github.com/htdt/self-supervised/blob/master/methods/whitening.py
    - https://github.com/roymiles/vkd/issues/2#issuecomment-2182980957
    """
    def __init__(self, features: torch.Tensor, eps: float = 0.0):
        """Whitening layer using Cholesky decomposition for 1D inputs.

        Args:
            feature_size (int): Dimensionality of each feature vector.
            eps (float, optional): Small value for numerical stability. Defaults to 0.0.
        """
        super().__init__()
        self.features_size = features.size(-1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computer centered and covariante matrix
        xn = x - x.mean(dim=0, keepdim=True)
        f_cov = torch.mm(xn.t(), xn) / (xn.size(0) - 1)

        # Add regularization and compute the inverse square root via Cholesky decomposition
        inv_sqrt = torch.linalg.cholesky((1 - self.eps) * f_cov + self.eps * torch.eye(self.feature_size, device=x.device))
        inv_sqrt = torch.inverse(inv_sqrt)

        # Apply the whitening transformation
        return torch.mm(xn, inv_sqrt)
