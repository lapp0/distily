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

    def __init__(self, student_features, teacher_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            teacher_features.size(-1),
            teacher_features.size(-1)
        ))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, student_features, teacher_features):
        # Enforce skew-symmetry on the weight matrix
        W_skew = (self.weight - self.weight.T) / 2
        # Apply matrix exponential to obtain an orthogonal matrix
        A = torch.linalg.matrix_exp(W_skew)
        # Apply the orthogonal projection to the student features
        projected_student_features = F.linear(student_features, A)

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
