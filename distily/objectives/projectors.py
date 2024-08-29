from functools import partial
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

    def __init__(self, student_features, teacher_features, use_batchnorm=False, use_layernorm=False):
        super().__init__()

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


PROJECTORS = {
    "identity": IdentityProjector,
    "linear": LinearProjector,
    "orthogonal": partial(OrthogonalProjector),

    # mlp
    "mlp": MLPProjector,
    "mlp_256_l2": partial(MLPProjector, hidden_dim=256, num_layers=2),
    "mlp_64_l2": partial(MLPProjector, hidden_dim=256, num_layers=2),
    "mlp_256_l3": partial(MLPProjector, hidden_dim=256, num_layers=3),
    "mlp_64_l3": partial(MLPProjector, hidden_dim=64, num_layers=3),
    "mlp_256_l4": partial(MLPProjector, hidden_dim=256, num_layers=4),
    "mlp_64_l4": partial(MLPProjector, hidden_dim=64, num_layers=4),
}
