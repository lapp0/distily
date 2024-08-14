import math
from functools import partial
from torch.nn import functional as F
import einops
from typing import List, Callable, Union, Tuple, Dict
from dataclasses import dataclass, fields
import torch
from transformers import PreTrainedModel


####################
# Distance Functions
####################
def _stable_kl_div(P_log_prob, Q_prob, epsilon=1e-10):
    """
    Stabilize by clamping Q_prob to avoid log(0) and division by zero
    """
    # ensure numerical stability
    Q_prob = Q_prob.clamp(min=epsilon)
    return F.kl_div(P_log_prob, Q_prob, reduction="none").sum(-1).mean()


def _cdist(x: torch.Tensor, y: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Builtin cdist only works for float32"""
    if x.dtype != torch.float32:
        # Reshape x and y for broadcasting
        x = einops.rearrange(x, "b l r -> b l () r")
        y = einops.rearrange(y, "b l r -> b () l r")
        # Compute the distance using the specified norm
        return (x - y).norm(dim=-1, p=p)
    # Use PyTorch's built-in cdist for other cases
    return torch.cdist(x, y, p=p)


def soft_mse_loss(feat_s, feat_t):
    student_prob = F.softmax(feat_s, dim=-1)
    teacher_prob = F.softmax(feat_t, dim=-1)
    return F.mse_loss(student_prob, teacher_prob)


def soft_mse_sum_loss(feat_s, feat_t):
    student_prob = F.softmax(feat_s, dim=-1)
    teacher_prob = F.softmax(feat_t, dim=-1)
    return F.mse_loss(student_prob, teacher_prob, reduction="none").sum(-1).mean()


def soft_cross_entropy_loss(feat_s, feat_t):
    student_prob = F.softmax(feat_s, dim=-1)
    teacher_prob = F.softmax(feat_t, dim=-1)
    return F.cross_entropy(student_prob, teacher_prob)


def kl_divergence_loss(feat_s, feat_t, epsilon=1e-10):
    student_log_prob = F.log_softmax(feat_s, dim=-1)
    teacher_prob = F.softmax(feat_t, dim=-1)
    return _stable_kl_div(student_log_prob, teacher_prob)


def reverse_kl_divergence_loss(feat_s, feat_t):
    teacher_log_prob = F.log_softmax(feat_t, dim=-1)
    student_prob = F.softmax(feat_s, dim=-1)
    return _stable_kl_div(teacher_log_prob, student_prob)


def cakld_loss(feat_s, feat_t, beta_prob=0.5):
    teacher_output_log_prob = F.log_softmax(feat_t, dim=-1)
    student_output_soft = F.softmax(feat_s, dim=-1)
    reverse_kl = _stable_kl_div(teacher_output_log_prob, student_output_soft)

    student_output_log_prob = F.log_softmax(feat_s, dim=-1)
    teacher_output_soft = F.softmax(feat_t, dim=-1)
    forward_kl = _stable_kl_div(student_output_log_prob, teacher_output_soft)

    kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
    return kl_loss


def jsd_loss(feat_s, feat_t, beta_prob=0.5):
    student_log_prob = F.log_softmax(feat_s, dim=-1)
    teacher_log_prob = F.log_softmax(feat_t, dim=-1)

    # Convert logits to probabilities for the mixed distribution calculation
    student_prob = student_log_prob.exp()
    teacher_prob = teacher_log_prob.exp()

    # Compute the mixed probability distribution
    m_prob = 0.5 * (student_prob + teacher_prob)

    # Calculate KL divergences between student/teacher log_probs and the mixed distribution
    kl_loss_f = _stable_kl_div(teacher_log_prob, m_prob)
    kl_loss_r = _stable_kl_div(student_log_prob, m_prob)

    # Compute the JSD as the average of the two KL divergences
    jsd = (kl_loss_f + kl_loss_r) / 2.0

    return jsd


def cosine_distance_loss(feat_s, feat_t):
    cosine_sim = F.cosine_similarity(feat_s, feat_t, dim=-1)
    cosine_distance = 1 - cosine_sim
    return cosine_distance.sum(-1).mean()


def mutual_information_loss(feat_s, feat_t, alpha=0.1):
    feat_s = feat_s.squeeze(1)
    feat_t = feat_t.squeeze(1)

    # TODO: this function doesn't work, fix or remove
    similarities = torch.matmul(feat_s, feat_t.T) / alpha

    # Create labels for the diagonal entries (correct matches)
    batch_size = feat_s.shape[0]
    labels = torch.arange(batch_size).to(feat_s.device)

    # cross entropy requires float32
    with torch.autocast(similarities.device.type):
        loss = F.cross_entropy(similarities, labels, reduction="none").sum(-1).mean()
    return loss


def sinkhorn_loss(feat_s, feat_t, epsilon=0.1, n_iters=20):
    """Based on algorithm in https://github.com/2018cx/SinKD/blob/main/loss.py#L119"""
    def sinkhorn_normalized(K, n_iters):
        for _ in range(n_iters):
            K = K / K.sum(dim=2, keepdim=True)
            K = K / K.sum(dim=1, keepdim=True)
        return K

    p_s = F.softmax(feat_s, dim=-1)
    p_t = F.softmax(feat_t, dim=-1)

    Wxy = _cdist(p_s, p_t, p=1)  # Cost Matrix
    K = torch.exp(-Wxy / epsilon)  # kernel matrix
    P = sinkhorn_normalized(K, n_iters)  # Sinkhorn iterations

    # EMD loss for batch
    return torch.sum(P * Wxy, dim=(1, 2)).mean()


LOSS_FUNCTIONS = {
    "kl": kl_divergence_loss,
    "mse": soft_mse_loss,
    "reverse_kl": reverse_kl_divergence_loss,
    "cakld": cakld_loss,
    "jsd": jsd_loss,
    "cos": cosine_distance_loss,
    "ce": soft_cross_entropy_loss,
    "mse_sum": soft_mse_sum_loss,

    # TODO: fix
    "mi": mutual_information_loss,
    "sinkhorn": sinkhorn_loss,

    # not recommended (TODO: delete?)
    "raw_mse": F.mse_loss,
    "raw_ce": F.cross_entropy,
}


###############
# Layer Mappers
###############
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


############
# Projectors
############
class LinearProjector:
    def __init__(self):
        self.W = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(num_layers, teacher_feature_size, student_feature_size)
            )
        )

    def apply(self, feat_s, feat_t):
        feat_s_projected = torch.einsum('lbsh,lhi->lbsi', feat_s, self.W)
        return feat_s_projected, feat_t


class SharedLinearProjector:
    """Only a single projector shared across all layers"""
    def __init__(self):
        self.W = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(teacher_feature_size, student_feature_size)
            )
        )

    def apply(self, feat_s, feat_t):
        feat_s_projected = torch.einsum('lbsh,hi->lbsi', feat_s, self.W)
        return feat_s_projected, feat_t


PROJECTORS = {
    "linear": LinearProjector,
    "shared_linear": SharedLinearProjector,
}


###########
# Objective
###########
@dataclass
class LossComponent:
    label: str
    weight: float
    loss_fn: Union[None, str, Callable]
    layer_mapper: Union[None, str, Callable] = None
    projector: Union[None, str, Callable] = None

    def _get_callable(self, attr, source_dict):
        if isinstance(attr, Union[str, None]):
            return source_dict[attr]
        if not callable(attr) and attr is not None:
            raise TypeError(f"{attr} must be a str, callable, or None")
        return attr

    @property
    def get_loss(self):
        return self._get_callable(self.loss_fn, LOSS_FUNCTIONS)

    @property
    def apply_layer_mapper(self):
        return self._get_callable(self.layer_mapper, LAYER_MAPPERS)

    @property
    def get_projector(self):
        return self._get_callable(self.projector, PROJECTORS)

    @property
    def is_measured(self):
        if bool(self.weight) != bool(self.loss_fn):
            raise ValueError(f"Expected both weight and loss_fn or neither, got {self.weight}, {self.loss_fn}")
        return bool(self.weight)

    def __repr__(self):
        class_name = self.__class__.__name__
        field_values = ', '.join(f"{field}={getattr(self, field)}" for field in self.__dataclass_fields__)
        return f"{class_name}({field_values})"


class DistillationObjective:
    """
    Comprehensive distillation objective to calculate loss based on various features.

    Implements __call__(teacher_model, student_model, inputs) -> loss

    Distillation loss can be calculated based on any number of model features, including
    - logits
    - specific hidden states (`hidden_states`)
    - attention scones (`attentions`)
    """
    def __init__(
            self,
            logits_weight,
            logits_loss_fn,
            hs_weight,
            hs_loss_fn,
            hs_layer_mapper,
            hs_projector,
            attn_weight,
            attn_loss_fn,
            attn_layer_mapper,
            attn_projector,
    ):
        self.logits_loss_component = LossComponent(
            "logits",
            weight=logits_weight,
            loss_fn=logits_loss_fn
        )
        self.hs_loss_component = LossComponent(
            "hs",
            weight=hs_weight,
            loss_fn=hs_loss_fn,
            layer_mapper=hs_layer_mapper,
        )
        self.attn_loss_component = LossComponent(
            "attn",
            weight=attn_weight,
            loss_fn=attn_loss_fn,
            layer_mapper=attn_layer_mapper,
        )

        self._projectors: dict = {}

    def __call__(self, teacher_model, student_model, inputs) -> Dict[str, float]:
        forward_kwargs = {
            **inputs,
            "output_hidden_states": self.hs_loss_component.is_measured,
            "output_attentions": self.attn_loss_component.is_measured,
        }
        # get student / teacher forward pass outputs
        with torch.no_grad():
            out_t = teacher_model(**forward_kwargs)
        out_s = student_model(**forward_kwargs)

        # calculate component loss
        device = student_model.device
        logits_loss = self._calc_loss(out_s.logits, out_t.logits, self.logits_loss_component, device)
        hs_loss = self._calc_loss(out_s.hidden_states, out_t.hidden_states, self.hs_loss_component, device)
        attn_loss = self._calc_loss(out_s.attentions, out_t.attentions, self.attn_loss_component, device)

        # calculate aggregate linear-combination loss
        loss = (
            logits_loss * self.logits_loss_component.weight +
            hs_loss * self.hs_loss_component.weight +
            attn_loss * self.attn_loss_component.weight
        )

        return {"loss": loss, "loss/logits": logits_loss, "loss/hs": hs_loss, "loss/attn": attn_loss}

    def _calc_loss(self, feat_s, feat_t, loss_component, device):
        if not loss_component.is_measured:
            return torch.tensor(0, device=device)

        if loss_component.projector:
            feat_s, feat_t = loss_component.apply_layer_mapper(feat_s, feat_t)
        elif isinstance(feat_s, tuple):
            feat_s, feat_t = torch.vstack(feat_s), torch.vstack(feat_t)

        if loss_component.projector:
            # projectors are trainable, therefore we lazy-load, then re-use the same projector
            projector = self._projectors.setdefault(loss_component.label, loss_component.get_projector())
            feat_s, feat_t = projector.apply(feat_s, feat_t)

        loss = loss_component.get_loss(feat_s, feat_t)

        return loss

    def __repr__(self):
        return (
            "DistillationObjective("
            f"logits_loss_component={self.logits_loss_component}, "
            f"hs_loss_component={self.hs_loss_component}, "
            f"attn_loss_component={self.attn_loss_component}"
            ")"
        )

        attrs = []
        for field in fields(self):
            value = getattr(self, field.name)
            # Check if the value is callable (a function, method, etc.)
            if callable(value):
                attrs.append(f"{field.name}=(fn:{value.__name__}())")
            else:
                attrs.append(f"{field.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"
