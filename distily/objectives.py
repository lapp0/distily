import collections
from torch.nn import functional as F
import einops
from typing import List, Callable, Union
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


def soft_mse_loss(student_features, teacher_features):
    student_prob = F.softmax(student_features, dim=-1)
    teacher_prob = F.softmax(teacher_features, dim=-1)
    return F.mse_loss(student_prob, teacher_prob)


def kl_divergence_loss(student_features, teacher_features, epsilon=1e-10):
    student_log_prob = F.log_softmax(student_features, dim=-1)
    teacher_prob = F.softmax(teacher_features, dim=-1)
    return _stable_kl_div(student_log_prob, teacher_prob)


def reverse_kl_divergence_loss(student_features, teacher_features):
    teacher_log_prob = F.log_softmax(teacher_features, dim=-1)
    student_prob = F.softmax(student_features, dim=-1)
    return _stable_kl_div(teacher_log_prob, student_prob)


def cakld_loss(student_features, teacher_features, beta_prob=0.5):
    teacher_output_log_prob = F.log_softmax(teacher_features, dim=-1)
    student_output_soft = F.softmax(student_features, dim=-1)
    reverse_kl = _stable_kl_div(teacher_output_log_prob, student_output_soft)

    student_output_log_prob = F.log_softmax(student_features, dim=-1)
    teacher_output_soft = F.softmax(teacher_features, dim=-1)
    forward_kl = _stable_kl_div(student_output_log_prob, teacher_output_soft)

    kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
    return kl_loss


def jsd_loss(student_features, teacher_features, beta_prob=0.5):
    student_prob = F.softmax(student_features, dim=-1)
    teacher_prob = F.softmax(teacher_features, dim=-1)

    c_prob = beta_prob * teacher_prob + (1 - beta_prob) * student_prob
    c_log_prob = c_prob.log()

    kl_loss_f = _stable_kl_div(c_log_prob, teacher_prob)
    kl_loss_r = _stable_kl_div(c_log_prob, student_prob)

    kl_loss = beta_prob * kl_loss_f + (1 - beta_prob) * kl_loss_r
    return kl_loss


def cosine_distance_loss(student_features, teacher_features):
    cosine_sim = F.cosine_similarity(student_features, teacher_features, dim=-1)
    cosine_distance = 1 - cosine_sim
    return cosine_distance.mean()


def mutual_information_loss(student_features, teacher_features, alpha=0.1):
    student_features = student_features.squeeze(1)
    teacher_features = teacher_features.squeeze(1)

    similarities = torch.matmul(student_features, teacher_features.T) / alpha

    # Create labels for the diagonal entries (correct matches)
    batch_size = student_features.shape[0]
    labels = torch.arange(batch_size).to(student_features.device)

    # cross entropy requires float32
    with torch.autocast(similarities.device.type):
        loss = F.cross_entropy(similarities, labels)
    return loss


def sinkhorn_loss(student_features, teacher_features, epsilon=0.1, n_iters=20):
    """Based on algorithm in https://github.com/2018cx/SinKD/blob/main/loss.py#L119"""
    def sinkhorn_normalized(K, n_iters):
        for _ in range(n_iters):
            K = K / K.sum(dim=2, keepdim=True)
            K = K / K.sum(dim=1, keepdim=True)
        return K

    p_s = F.softmax(student_features, dim=-1)
    p_t = F.softmax(teacher_features, dim=-1)

    Wxy = _cdist(p_s, p_t, p=1)  # Cost Matrix
    K = torch.exp(-Wxy / epsilon)  # kernel matrix
    P = sinkhorn_normalized(K, n_iters)  # Sinkhorn iterations

    # EMD loss for batch
    return torch.sum(P * Wxy, dim=(1, 2)).mean()


"""
Recommendations for loss functions, in order of expected performance.

Activations:
- MI-α
- MSE
- PKD
- CE

Logits:
- KL
- No other loss function is close to its performance, experimentally

Attentions:
(todo experiment)
- - Cos
- sort: MI-α, MSE, KL, Reverse KL, JSD, CAKLD

TODO CATEGORIZE:
- sinkhorn

"""
LOSS_FUNCTIONS = {
    "mi": mutual_information_loss,
    "mse": soft_mse_loss,
    "kl": kl_divergence_loss,
    "reverse_kl": reverse_kl_divergence_loss,
    "cakld": cakld_loss,
    "jsd": jsd_loss,
    "cos": cosine_distance_loss,
    "sinkhorn": sinkhorn_loss,
    "ce": F.cross_entropy,

    # not recommended (TODO: delete?)
    "raw_mse": F.mse_loss,
}


###############
# Layer Mappers
###############
def full_layer_mapper(student_features, teacher_features):
    assert student_features.shape == teacher_features.shape
    return student_features, teacher_features


def skip_layer_mapper(student_features, teacher_features):
    raise NotImplementedError


def last_layer_mapper(student_features, teacher_features):
    raise NotImplementedError


def emd_layer_mapper(student_features, teacher_features):
    raise NotImplementedError


LAYER_MAPPERS = {
    "full": full_layer_mapper,
    "skip": skip_layer_mapper,
    "last": last_layer_mapper,
    "emd": emd_layer_mapper,
}


#####################
# Objective Functions
#####################
class DistillationObjective:
    """
    Callable to calculate distillation loss.

    Implements __call__(teacher_model, student_model, inputs) -> loss

    Distillation loss can be calculated based on any number of model features, including
    - logits
    - specific activations (`hidden_states`)
    - attention scones (`attentions`)
    """
    required_equivalent_config: List[str]
    loss_fn: Callable

    def __init__(self, teacher_config, student_config):
        self._assert_required_configurations_equivalent(teacher_config, student_config)
        self.teacher_config = teacher_config
        self.student_config = student_config

    def __call__(
        self,
        teacher_model: PreTrainedModel,
        student_model: PreTrainedModel,
        inputs,
    ):
        ...

    @classmethod
    def _assert_required_configurations_equivalent(cls, teacher_config, student_config):
        """
        Helper function to assert that the required_equivalent_config are equivalent
        """
        mismatched = []
        for config_key in cls.required_equivalent_config:
            if teacher_config.to_dict().get(config_key) != student_config.to_dict().get(config_key):
                mismatched.append((
                    config_key,
                    teacher_config[config_key],
                    student_config[config_key]
                ))
        if mismatched:
            mismatch_configs_formatted = "\n".join([
                f"Config Key: {config_key}, teacher value: {teacher_val}, student value: {student_val}"
                for config_key, teacher_val, student_val in mismatched
            ])
            raise ValueError(
                f"{cls.__name__} requires models to have the following equivalent configurations:\n"
                f"{cls.required_equivalent_config}\n"
                "The following configurations didn't match:\n"
                f"{mismatch_configs_formatted}"
            )


class LogitsObjective(DistillationObjective):
    """
    logits of forward pass
    """
    def __init__(self, loss_fn: Union[str, Callable] = "mse"):
        if isinstance(loss_fn, str):
            loss_fn = LOSS_FUNCTIONS[loss_fn]
        self.loss_fn = loss_fn

    def __call__(self, teacher_model, student_model, inputs):
        with torch.no_grad():
            teacher_features = teacher_model(**inputs)
        student_features = student_model(**inputs)

        return self.loss_fn(student_features.logits, teacher_features.logits)


class ActivationsObjective(DistillationObjective):
    """
    hidden states of forward pass
    """
    def __init__(self, loss_fn: Callable = "mse"):
        if isinstance(loss_fn, str):
            loss_fn = LOSS_FUNCTIONS[loss_fn]
        self.loss_fn = loss_fn

    def __call__(self, teacher_model, student_model, inputs):
        with torch.no_grad():
            teacher_features = teacher_model(**inputs, output_hidden_states=True)
        student_features = student_model(**inputs, output_hidden_states=True)

        return self.loss_fn(student_features.hidden_states, teacher_features.hidden_states)


class AttentionsObjective(DistillationObjective):
    """
    attentions of forward pass
    """
    def __init__(self, loss_fn: Callable = "mse"):
        if isinstance(loss_fn, str):
            loss_fn = LOSS_FUNCTIONS[loss_fn]
        self.loss_fn = loss_fn

    def __call__(self, teacher_model, student_model, inputs):
        with torch.no_grad():
            teacher_features = teacher_model(**inputs, output_attentions=True)
        student_features = student_model(**inputs, output_attentions=True)

        return self.loss_fn(student_features.attentions, teacher_features.attentions)


class LegacyObjective(DistillationObjective):
    # Hard coded, to reproduce old success
    def __init__(self, loss_fn: Callable = "kl"):
        self.loss_fn = kl_divergence_loss

    def __call__(self, teacher_model, student_model, inputs):
        with torch.no_grad():
            teacher_features = teacher_model(**inputs, output_hidden_states=True)
        student_features = student_model(**inputs, output_hidden_states=True)

        logits_loss = self.loss_fn(student_features.logits, teacher_features.logits)
        activations_loss = self.loss_fn(
            torch.stack(student_features.hidden_states),
            torch.stack(teacher_features.hidden_states)
        )

        # legacy, this is an incorrect implementation:
        return logits_loss + activations_loss * len(student_features.hidden_states)


@dataclass
class MultiObjective(DistillationObjective):
    """
    Comprehensive distillation objective to calculate loss based on various features.
    Implements __call__(teacher_model, student_model, inputs) -> loss
    """
    logits_weight: float = 1
    logits_loss_fn: Union[None, str, Callable] = "kl"
    activations_weight: float = 0
    activations_loss_fn: Union[None, str, Callable] = "mse"
    attentions_weight: float = 0
    attentions_loss_fn: Union[None, str, Callable] = "mse"

    def __post_init__(self):
        if isinstance(self.logits_loss_fn, str):
            self.logits_loss_fn = LOSS_FUNCTIONS[self.logits_loss_fn]
        if isinstance(self.activations_loss_fn, str):
            self.activations_loss_fn = LOSS_FUNCTIONS[self.activations_loss_fn]
        if isinstance(self.attentions_loss_fn, str):
            self.attentions_loss_fn = LOSS_FUNCTIONS[self.attentions_loss_fn]

    def __call__(self, teacher_model, student_model, inputs):
        forward_kwargs = {
            **inputs,
            "output_hidden_states": (self.activations_weight != 0),
            "output_attentions": (self.attentions_weight != 0)
        }
        with torch.no_grad():
            teacher_outputs = teacher_model(**forward_kwargs)
        student_outputs = student_model(**forward_kwargs)

        losses = {
            "loss/logits": self._get_logit_loss(student_outputs, teacher_outputs),
            "loss/activations": self._get_activation_loss(teacher_outputs, student_outputs),
            "loss/attentions": self._get_attentions_loss(teacher_outputs, student_outputs),
        }

        losses["loss"] = losses["loss/logits"] + losses["loss/activations"]
        return losses

    def _get_logit_loss(self, student_outputs, teacher_outputs):
        if not self.logits_weight:
            return torch.tensor(0, device=student_outputs.logits.device)
        return self.logits_loss_fn(
            student_outputs.logits, teacher_outputs.logits
        ) * self.logits_weight

    def _get_activation_loss(self, teacher_outputs, student_outputs):
        if not self.activations_weight:
            return torch.tensor(0, device=student_outputs.logits.device)
        return self.activations_loss_fn(
            torch.stack(student_outputs.hidden_states),
            torch.stack(teacher_outputs.hidden_states),
        ) * self.activations_weight

    def _get_attentions_loss(self, teacher_outputs, student_outputs):
        if not self.activations_weight:
            return torch.tensor(0, device=student_outputs.logits.device)
        raise NotImplementedError

    def __repr__(self):
        attrs = []
        for field in fields(self):
            value = getattr(self, field.name)
            # Check if the value is callable (a function, method, etc.)
            if callable(value):
                attrs.append(f"{field.name}=(fn:{value.__name__}())")
            else:
                attrs.append(f"{field.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"


OBJECTIVES = {
    "legacy": LegacyObjective,  # TODO: remove
    "logits": LogitsObjective,
    "multi": MultiObjective,
}
