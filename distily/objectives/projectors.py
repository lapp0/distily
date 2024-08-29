from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from distily.objectives.norm import BatchNorm1d4d, LayerNorm1d4d


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

    def __init__(self, student_features, teacher_features, use_batchnorm=False, use_layernorm=False):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            bn_s = BatchNorm1d4d(teacher_features.size(-1), eps=0.0001, affine=False)
            bn_t = BatchNorm1d4d(teacher_features.size(-1), eps=0.0001, affine=False)
            self.register_module('bn_s', bn_s)
            self.register_module('bn_t', bn_t)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            ln_s = LayerNorm1d4d(teacher_features.size(-1), eps=0.0001, elementwise_affine=False)
            ln_t = LayerNorm1d4d(teacher_features.size(-1), eps=0.0001, elementwise_affine=False)
            self.register_module('ln_s', ln_s)
            self.register_module('ln_t', ln_t)

        teacher_dim = teacher_features.size(-1)
        self.student_dim = student_features.size(-1)

        # Ensure skew-symmetry
        self.weight = nn.Parameter(torch.empty((teacher_dim, teacher_dim)))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, student_features, teacher_features):
        W = (self.weight - self.weight.T) / 2  # Enforcing skew symmetry
        A = torch.linalg.matrix_exp(W.float()).to(dtype=W.dtype)  # mat exp - float32 for numerical stability
        A = A[:, 0:self.student_dim]
        projected_student_features = F.linear(student_features, A)

        if self.use_batchnorm:
            projected_student_features = self.bn_s(projected_student_features)
            teacher_features = self.bn_t(teacher_features)

        elif self.use_layernorm:
            projected_student_features = self.ln_s(projected_student_features)
            teacher_features = self.ln_t(teacher_features)

        return projected_student_features, teacher_features


class MLPProjector(nn.Module):
    """Applies a multi-layer perceptron (MLP) transformation to student features."""

    def __init__(self, student_features, teacher_features, hidden_dim=256, num_layers=2):
        super().__init__()
        in_features = student_features.size(-1)
        out_features = teacher_features.size(-1)

        layers = [nn.Linear(in_features, hidden_dim), nn.ReLU()]
        layers += [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers - 2)]
        layers.append(nn.Linear(hidden_dim, out_features))

        self.proj = nn.Sequential(*layers)

    def forward(self, student_features, teacher_features):
        return self.proj(student_features), teacher_features


# TODO: update EnsembleProjector so it accepts arbitrary child projectors, not just linear
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
            # apply 1d batchnorm on 4d tensor with layers coupled
            student_projected = self.bn_s(
                student_projected.reshape(-1, student_projected.size(-1))
            ).reshape_as(student_projected)
            teacher_features = self.bn_t(
                teacher_features.reshape(-1, teacher_features.size(-1))
            ).reshape_as(teacher_features)
        elif self.use_decoupled_batchnorm:
            # layer-decoupled batchnorm
            # TODO: enable and test
            for i in range(student_projected.size(0)):  # Iterate over num_layers
                student_projected[i] = self.bn_s(
                    student_projected[i].reshape(-1, student_projected[i].size(-1))
                ).reshape_as(student_projected[i])

            for i in range(teacher_features.size(0)):  # Iterate over num_layers
                teacher_features[i] = self.bn_t(
                    teacher_features[i].reshape(-1, teacher_features[i].size(-1))
                ).reshape_as(teacher_features[i])

        return student_projected, teacher_features


PROJECTORS = {
    "identity": IdentityProjector,
    "linear": LinearProjector,
    "miles": MilesProjector,

    "orthogonal": partial(OrthogonalProjector),
    "orthogonal_batchnorm": partial(OrthogonalProjector, use_batchnorm=True),
    "orthogonal_layernorm": partial(OrthogonalProjector, use_layernorm=True),

    # mlp
    "mlp": MLPProjector,
    "mlp_256_l2": partial(MLPProjector, hidden_dim=256, num_layers=2),
    "mlp_64_l2": partial(MLPProjector, hidden_dim=256, num_layers=2),
    "mlp_256_l3": partial(MLPProjector, hidden_dim=256, num_layers=3),
    "mlp_64_l3": partial(MLPProjector, hidden_dim=64, num_layers=3),
    "mlp_256_l4": partial(MLPProjector, hidden_dim=256, num_layers=3),
    "mlp_64_l4": partial(MLPProjector, hidden_dim=64, num_layers=3),
}
