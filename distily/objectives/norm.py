import torch.nn as nn


class FourDimDistillationNorm:
    def forward(self, student_features, teacher_features):
        """
        Flatten first three dimensions to a single dimension,
        apply norm to both student and teacher
        """
        assert student_features.shape == teacher_features.shape
        assert len(student_features.shape) == 4

        batch_size, layers, sequence_length, feature_size = student_features.shape

        s_flattened = student_features.view(batch_size * layers * sequence_length, feature_size)
        s_flattened_norm = self.norm_s(s_flattened)
        s_norm = s_flattened_norm.view(batch_size, layers, sequence_length, feature_size)

        t_flattened = teacher_features.view(batch_size * layers * sequence_length, feature_size)
        t_flattened_norm = self.norm_t(t_flattened)
        t_norm = t_flattened_norm.view(batch_size, layers, sequence_length, feature_size)

        return s_norm, t_norm


class DistillationBatchNorm1d(nn.Module):
    """
    Custom BatchNorm that accepts a 4D tensor of shape
    (batch size, num layers, sequence length, feature size)
    and applies batch normalization
    """
    def __init__(self, student_features, teacher_features, eps=1e-5, affine=False, **kwargs):
        super().__init__()
        size_t = teacher_features.size(-1)
        self.norm_s = nn.BatchNorm1d(size_t, eps=eps, affine=affine, **kwargs)
        self.norm_t = nn.BatchNorm1d(size_t, eps=eps, affine=affine, **kwargs)

    forward = FourDimDistillationNorm.forward


class DistillationLayerNorm1d(nn.LayerNorm):
    """
    Calculate layernorm across 4D tensor of shape
    (batch size, num layers, sequence length, feature size)
    """
    def __init__(self, student_features, teacher_features, eps=1e-5, elementwise_affine=False, **kwargs):
        super().__init__()
        size_t = teacher_features.size(-1)
        self.norm_s = nn.LayerNorm(size_t, eps=eps, elementwise_affine=elementwise_affine, **kwargs)
        self.norm_t = nn.LayerNorm(size_t, eps=eps, elementwise_affine=elementwise_affine, **kwargs)

    forward = FourDimDistillationNorm.forward


NORMS = {
    "batchnorm": DistillationBatchNorm1d,
    "layernorm": DistillationLayerNorm1d,
}
