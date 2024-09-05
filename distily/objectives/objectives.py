from typing import Callable, Union, Dict
from dataclasses import dataclass
import torch

from distily.objectives import loss, layer_mappers, norm, projectors


@dataclass
class LossComponent:
    label: str
    weight: float
    loss_fn: Union[None, str, Callable]
    layer_mapper: Union[None, str, Callable] = None
    norm: Union[None, str, Callable] = None
    projector: Union[None, str, Callable] = None

    def _get_callable(self, attr, source_dict):
        if isinstance(attr, Union[str, None]):
            return source_dict[attr]
        if not callable(attr) and attr is not None:
            raise TypeError(f"{attr} must be a str, callable, or None")
        return attr

    @property
    def get_loss(self):
        return self._get_callable(self.loss_fn, loss.LOSS_FUNCTIONS)

    @property
    def apply_layer_mapper(self):
        return self._get_callable(self.layer_mapper, layer_mappers.LAYER_MAPPERS)

    @property
    def get_norm(self):
        return self._get_callable(self.norm, norm.NORMS)

    @property
    def get_projector(self):
        return self._get_callable(self.projector, projectors.PROJECTORS)

    @property
    def is_measured(self):
        return bool(self.weight)

    def __repr__(self):
        prefix = "\n    "

        if not self.is_measured:
            return f"{self.__class__.__name__}({prefix}weight=0\n)"

        field_values = ','.join(
            f"{prefix}{field}={repr(getattr(self, field))}"
            for field in self.__dataclass_fields__
            if getattr(self, field) is not None
        )
        return f"{self.__class__.__name__}({field_values}\n)"


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
            hs_norm,
            hs_projector,
            attn_weight,
            attn_loss_fn,
            attn_layer_mapper,
            attn_norm,
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
            norm=hs_norm,
            projector=hs_projector,
        )
        self.attn_loss_component = LossComponent(
            "attn",
            weight=attn_weight,
            loss_fn=attn_loss_fn,
            layer_mapper=attn_layer_mapper,
            norm=attn_norm,
            projector=attn_projector,
        )

        self._projectors: dict = {}
        self._norms: dict = {}

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

        if loss_component.layer_mapper:
            feat_s, feat_t = loss_component.apply_layer_mapper(feat_s, feat_t)
        elif isinstance(feat_s, tuple):
            feat_s, feat_t = torch.vstack(feat_s), torch.vstack(feat_t)

        # projectors and norms may be trainable, therefore we lazy-load, then re-use
        if loss_component.projector:
            if loss_component.label not in self._projectors:
                self._projectors[loss_component.label] = loss_component\
                    .get_projector(feat_s, feat_t)\
                    .to(device=feat_s.device, dtype=feat_s.dtype)
            feat_s, feat_t = self._projectors[loss_component.label].forward(feat_s, feat_t)

        if loss_component.norm:
            if loss_component.label not in self._norms:
                self._norms[loss_component.label] = loss_component\
                    .get_norm(feat_s, feat_t)\
                    .to(device=feat_s.device, dtype=feat_s.dtype)
            feat_s, feat_t = self._norms[loss_component.label].forward(feat_s, feat_t)

        loss = loss_component.get_loss(feat_s, feat_t)

        return loss

    def __repr__(self):
        components = [
            f"logits_loss_component={self.logits_loss_component}",
            f"hs_loss_component={self.hs_loss_component}",
            f"attn_loss_component={self.attn_loss_component}"
        ]
        prefix = "\n    "
        components = prefix + f",{prefix}".join([c.replace("\n", prefix) for c in components])
        return f"{self.__class__.__name__}({components}\n)"
