import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityProjector(nn.Module):
    """Returns student features unchanged."""

    def __init__(self, student_features, teacher_features):
        super().__init__()

    def forward(self, student_features, teacher_features):
        return student_features, teacher_features


class LinearProjector(nn.Module):
    """Applies a linear transformation to student features."""

    def __init__(self, student_features, teacher_features):
        super().__init__()
        self.proj = nn.Linear(
            student_features.size(-1),
            teacher_features.size(-1)
        )

    def forward(self, student_features, teacher_features):
        return self.proj(student_features), teacher_features


class OrthogonalProjector(nn.Module):
    """
    Applies an orthogonal transformation to student features.
    Based on paper: https://arxiv.org/abs/2403.06213
    Based on: https://github.com/roymiles/vkd/issues/1#issuecomment-2135090288
    """

    def __init__(self, student_features, teacher_features, pade_approx=False):
        super().__init__()

        if pade_approx:
            raise NotImplementedError("Pade Approximation is not implemented")

        teacher_dim = teacher_features.size(-1)
        self.student_dim = student_features.size(-1)

        weight = torch.empty((teacher_dim, teacher_dim))
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Ensure skew-symmetry
        self.weight = nn.Parameter((weight - weight.T) / 2)

    def forward(self, student_features, teacher_features):
        A = torch.linalg.matrix_exp(self.weight)

        if self.student_dim != A.size(0):
            # Truncate A to match the student dimension
            A = A[:self.student_dim, :]
            # Apply QR decomposition to project onto the Stiefel manifold
            Q, _ = torch.linalg.qr(A)
        else:
            Q = A

        projected_student_features = F.linear(student_features, Q)

        return projected_student_features, teacher_features


class MLPProjector(nn.Module):
    """Applies a multi-layer perceptron (MLP) transformation to student features."""

    def __init__(self, student_features, teacher_features, hidden_dim=256):
        super().__init__()
        in_features = student_features.size(-1)
        out_features = teacher_features.size(-1)
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, student_features, teacher_features):
        return self.proj(student_features), teacher_features


class EnsembleProjector(nn.Module):
    """Applies an ensemble of linear projections to student features."""

    def __init__(self, student_features, teacher_features, num_projectors=3):
        super().__init__()
        self.proj = nn.ParameterList([
            nn.Linear(
                student_features.size(-1),
                teacher_features.size(-1),
            )
            for _ in range(num_projectors)
        ])

    def forward(self, student_features, teacher_features):
        outputs = torch.stack([proj(student_features) for proj in self.proj], dim=0)
        return torch.mean(outputs, dim=0), teacher_features


class MilesProjector(nn.Module):
    """Applies projector based on paper https://arxiv.org/pdf/2303.11098"""
    def __init__(self, student_features, teacher_features, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        # Define the linear projection layer
        self.proj = nn.Linear(
            student_features.size(-1),
            teacher_features.size(-1),
        )

        if use_batchnorm:
            bn_s = nn.BatchNorm1d(teacher_features.size(-1), eps=0.0001, affine=False)
            bn_t = nn.BatchNorm1d(teacher_features.size(-1), eps=0.0001, affine=False)
            self.register_module('bn_s', bn_s)
            self.register_module('bn_t', bn_t)

    def forward(self, student_features, teacher_features):
        student_projected = self.proj(student_features)

        if self.use_batchnorm:
            student_projected = self.bn_s(student_projected)
            teacher_features = self.bn_t(teacher_features)

        return student_projected, teacher_features


PROJECTORS = {
    "identity": IdentityProjector,
    "linear": LinearProjector,
    "orthogonal": OrthogonalProjector,
    "mlp": MLPProjector,
    "ensemble": EnsembleProjector,
    "miles": MilesProjector,
}
