from typing import Callable, Union, Dict
from dataclasses import dataclass
import torch
from torch import nn

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
        # TODO: rewrite loss.py to use nn.Module, not functional
        return self._get_callable(self.loss_fn, loss.LOSS_FUNCTIONS)(*a, **kw)

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
            if field not in ("label", "_cache") and getattr(self, field) is not None
        )
        return f"{self.__class__.__name__}({field_values}\n)"


class LazyDistillationLossPipeline(nn.modules.lazy.LazyModuleMixin, nn.Module):
    def __init__(self, loss_component):
        super().__init__()

        # not modules, parameters or functions
        self.weight = loss_component.weight
        self.loss_fn = loss_component.get_loss  # TODO: use loss module, not functional
        self.layer_mapper_fn = loss_component.apply_layer_mapper

        # store the module class for lazy initialization
        self._projector_cls = loss_component.get_projector  # TODO: better name
        self._norm_cls = loss_component.get_norm  # TODO: better name

        # uninitialized modules
        self.projector = nn.parameter.UninitializedParameter()
        self.norm = nn.parameter.UninitializedParameter()

    @torch.no_grad()
    def initialize_parameters(self, feat_s, feat_t):
        if not self.has_uninitialized_params():
            return

        device, dtype = feat_s.device, feat_s.dtype

        if isinstance(self.projector, nn.parameter.UninitializedParameter):
            projector_module = self._projector_cls(feat_s, feat_t).to(device=device, dtype=dtype)
            self.projector.materialize(projector_module)

        if isinstance(self.norm, nn.parameter.UninitializedParameter):
            norm_module = self._norm_cls(feat_s, feat_t).to(device=device, dtype=dtype)
            self.norm.materialize(norm_module)

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        if not self.weight:
            return torch.tensor(0, device=feat_s.device)

        if self.layer_mapper_fn:
            feat_s, feat_t = self.layer_mapper_fn(feat_s, feat_t)
        elif isinstance(feat_s, tuple):
            feat_s, feat_t = torch.vstack(feat_s), torch.vstack(feat_t)

        feat_s, feat_t = self.projector(feat_s, feat_t)
        feat_s, feat_t = self.norm(feat_s, feat_t)
        loss = self.loss_fn(feat_s, feat_t)

        return loss * self.weight


class DistillationObjective:
    """
    Comprehensive distillation objective to calculate loss based on various features.

    Implements __call__(teacher_model, student_model, inputs) -> loss

    Mechanism
    ---------
    Runs forward pass and retrieves forward pass features
    - `out_s = student_model.forward()` (with gradients)
    - `out_t = teacher_model.forward()` (WITHOUT gradients)

    Then applies a loss function, `return loss(out_s, out_t)`

    Forward Pass Features
    ---------------------
    attentions:
      - Tuple shape: (num_layers,)
      - Tensor shape: (batch_size, num_attention_heads, sequence_length, sequence_length)
      - Contains attention scores for each layer.

    hidden_states:
      - Tuple shape: (num_layers + 1,)
      - Tensor shape: (batch_size, sequence_length, hidden_state_size)
      - Represents hidden states for each layer and the initial embedding.

    past_key_values:
      - Tuple shape: (num_layers, 2,)
      - Tensor shape: (batch_size, num_attention_heads, sequence_length, embedding_size_per_head)
      - keys and values (tuple of 2) for faster decoding.


    First Layer Data Flow
    ---------------------
    Embedding (hidden_states[0])
    -> MHA (attentions[0], past_key_values[0])
    -> FFN (updated hidden_states[1])
    -> Next Layer
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

        self.logits_loss_fn = LazyDistillationLossPipeline(self.logits_loss_component)
        self.hs_loss_fn = LazyDistillationLossPipeline(self.hs_loss_component)
        self.attn_loss_fn = LazyDistillationLossPipeline(self.attn_loss_component)

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

        loss_logits = self.logits_loss_fn(out_s.logits, out_t.logits)
        loss_hs = self.hs_loss_fn(out_s.hidden_states, out_t.hidden_states)
        loss_attn = self.attn_loss_fn(out_s.attentions, out_t.attentions)

        loss = loss_logits + loss_hs + loss_attn

        return {"loss": loss, "loss/logits": loss_logits, "loss/hs": loss_hs, "loss/attn": loss_attn}

    def __repr__(self):
        components = [
            f"logits_loss_component={self.logits_loss_component}",
            f"hs_loss_component={self.hs_loss_component}",
            f"attn_loss_component={self.attn_loss_component}"
        ]
        prefix = "\n    "
        components = prefix + f",{prefix}".join([c.replace("\n", prefix) for c in components])
        return f"{self.__class__.__name__}({components}\n)"
