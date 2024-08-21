import torch
import math
from functools import partial

from typing import List, Tuple


def index_layer_mapper(feat_s, feat_t, index_mapper: List[Tuple[int, int]]):
    """
    Maps specified student layers to corresponding teacher layers.

    Args:
        feat_s: Student feature tensor.
        feat_t: Teacher feature tensor.
        index_mapper: List of (student_layer, teacher_layer) index pairs.

    Returns:
        Mapped student and teacher tensors.
    """
    student_mapped = torch.stack([feat_s[i] for i, _ in index_mapper])
    teacher_mapped = torch.stack([feat_t[j] for _, j in index_mapper])

    return student_mapped, teacher_mapped


def sequential_layer_mapper(feat_s, feat_t, start, end):
    """
    Maps student layers to teacher layers sequentially from start_layer to end_layer.
    input and output shape: (num layers, batch size, sequence length, feature size)
    """
    feat_t = feat_t[start:end]
    feat_s = feat_s[start:end]
    return torch.stack(feat_s), torch.stack(feat_t)


def single_layer_mapper(feat_s, feat_t, layer):
    end_idx = (layer, layer + 1) if layer != -1 else (-1, None)
    return sequential_layer_mapper(feat_s, feat_t, start=layer, end=end_idx)


def last_k_layers_mapper(feat_s, feat_t, num_layers):
    return sequential_layer_mapper(feat_s, feat_t, start=(-num_layers))


def uniform_consecutive_layer_mapper(feat_s, feat_t):
    num_student_layers = feat_s.size(0)
    num_teacher_layers = feat_t.size(0)
    k = math.ceil(num_teacher_layers / num_student_layers)

    index_mapper = []
    for i in range(num_student_layers):
        start = k * i
        end = min(k * (i + 1), num_teacher_layers)
        index_mapper.extend([(i, j) for j in range(start, end)])
    return index_layer_mapper(feat_s, feat_t, index_mapper)


def uniform_last_layer_mapper(feat_s, feat_t):
    num_student_layers = feat_s.size(0)
    num_teacher_layers = feat_t.size(0)

    index_mapper = []
    for i in range(num_student_layers):
        uniform_layer = i * num_teacher_layers // num_student_layers
        index_mapper.append((i, uniform_layer))
        index_mapper.append((i, -1))  # Adding last layer mapping
    return index_layer_mapper(feat_s, feat_t, index_mapper)


LAYER_MAPPERS = {
    "all": partial(sequential_layer_mapper, start=None, end=None),
    "last": partial(single_layer_mapper, layer=-1),
    "last_k_2": partial(last_k_layers_mapper, num_layers=2),
    "last_k_3": partial(last_k_layers_mapper, num_layers=3),
    "last_k_4": partial(last_k_layers_mapper, num_layers=4),
    "layer-2": partial(single_layer_mapper, layer=-2),
    "layer-3": partial(single_layer_mapper, layer=-3),
    "layer-4": partial(single_layer_mapper, layer=-4),
    "uniform_cons": uniform_consecutive_layer_mapper,
    "uniform+last": uniform_last_layer_mapper,
}
