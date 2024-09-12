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

    def __init__(self, student_features, teacher_features):
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

    def __init__(self, student_features, teacher_features, hidden_dim=64):
        super().__init__()
        in_features = student_features.size(-1)
        out_features = teacher_features.size(-1)

        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, student_features, teacher_features):
        for name, param in self.proj.named_parameters():
            print(f"module hash: {hash(self)} | parameter: {name} | Grad: {param.requires_grad} | Shape: {param.shape} | Min {param.min()} | Median {param.median()}")
        return self.proj(student_features), teacher_features


# TODO: update EnsembleProjector
# - https://arxiv.org/pdf/2210.15274.pdf
# - https://github.com/chenyd7/PEFD

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
    None: IdentityProjector,
    "linear": LinearProjector,
    "orthogonal": partial(OrthogonalProjector),

    # mlp
    "mlp": MLPProjector,
    "mlp_256": partial(MLPProjector, hidden_dim=256),
    "mlp_64": partial(MLPProjector, hidden_dim=64),
}
