import torch.nn as nn
from functools import partial


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
    def __init__(self, student_feat, teacher_feat, norm_student=True, affine=False, **kwargs):
        super().__init__()
        size_t = teacher_feat.size(-1)
        if norm_student:
            self.norm_s = nn.BatchNorm1d(size_t, affine=affine, **kwargs)
        else:
            self.norm_s = nn.Identity()
        self.norm_t = nn.BatchNorm1d(size_t, affine=affine, **kwargs)

    forward = FourDimDistillationNorm.forward


class DistillationLayerNorm1d(nn.Module):
    """
    Calculate layernorm across 4D tensor of shape
    (batch size, num layers, sequence length, feature size)
    """
    def __init__(self, student_feat, teacher_feat, norm_student=True, affine=False, **kwargs):
        super().__init__()
        size_t = teacher_feat.size(-1)
        if norm_student:
            self.norm_s = nn.LayerNorm(size_t, elementwise_affine=affine, **kwargs)
        else:
            self.norm_s = nn.Identity()
        self.norm_t = nn.LayerNorm(size_t, elementwise_affine=affine, **kwargs)

    forward = FourDimDistillationNorm.forward


class DistillationRMSNorm1d(nn.Module):
    def __init__(self, student_feat, teacher_feat, norm_student=True, affine=False, **kwargs):
        super().__init__()
        size_t = teacher_feat.size(-1)
        if norm_student:
            self.norm_s = nn.RMSNorm(size_t, elementwise_affine=affine, **kwargs)
        else:
            self.norm_s = nn.Identity()
        self.norm_t = nn.RMSNorm(size_t, elementwise_affine=affine, **kwargs)

    forward = FourDimDistillationNorm.forward


class DistillationInstanceNorm1d(nn.Module):
    def __init__(self, student_feat, teacher_feat, norm_student=True, affine=False, **kwargs):
        super().__init__()
        # InstanceNorm1d uses the zeroth dim's size, we lazy-set this in forward()
        if norm_student:
            self.norm_s = nn.LazyInstanceNorm1d(affine=affine, **kwargs)
        else:
            self.norm_s = nn.Identity()
        self.norm_t = nn.LazyInstanceNorm1d(affine=affine, **kwargs)

    forward = FourDimDistillationNorm.forward


# TODO:
# class DistillationGroupNorm1d(nn.Module):


NORMS = {
    "batchnorm": DistillationBatchNorm1d,
    "layernorm": DistillationLayerNorm1d,
    "rmsnorm": DistillationRMSNorm1d,
    "instancenorm": DistillationInstanceNorm1d,
}

# apply all permutations of `teacher_only`, `affine`, and `track_running_stats`
NORMS.update({
    f"{norm_name}_teacher_only": partial(norm_fn, norm_student=False)
    for norm_name, norm_fn in NORMS.items()
})
NORMS.update({
    f"{norm_name}_affine": partial(norm_fn, affine=True)
    for norm_name, norm_fn in NORMS.items()
})
NORMS.update({
    f"{norm_name}_stats": partial(norm_fn, track_running_stats=True)
    for norm_name, norm_fn in NORMS.items()
    if norm_name.startswith("batchnorm") or norm_name.startswith("instancenorm")
})
