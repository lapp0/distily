import torch.nn as nn
from functools import partial


class DistillationIdentityNorm(nn.Module):
    def __init__(self, student_features, teacher_features, **kwargs):
        super().__init__()

    def forward(self, student_features, teacher_features, **kwargs):
        return student_features, teacher_features


class DistillationLayerNorm(nn.Module):
    def __init__(self, student_feat, teacher_feat, norm_student=True, affine=False, **kwargs):
        super().__init__()
        self.teacher_norm = nn.LayerNorm(teacher_feat.shape[-1:], elementwise_affine=affine, **kwargs)
        if norm_student:
            self.student_norm = nn.LayerNorm(student_feat.shape[-1:], elementwise_affine=affine, **kwargs)
        else:
            self.student_norm = nn.Identity()

    def forward(self, student_features, teacher_features):
        teacher_features = self.teacher_norm(teacher_features)
        student_features = self.student_norm(student_features)
        return student_features, teacher_features


# TODO: fix other norms, layernorm is the only one which is correct right now though
# Old norm implementations existed in 6cb53a8

NORMS = {
    None: DistillationIdentityNorm,
    #"batchnorm": DistillationBatchNorm1d,
    "layernorm": DistillationLayerNorm,# DistillationLayerNorm1d,
    #"rmsnorm": DistillationRMSNorm1d,

    # TODO: fix bugs
    # "instancenorm": DistillationInstanceNorm1d,
    # "groupnorm": TODO
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
