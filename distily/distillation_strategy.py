from typing import List, NamedTuple, Dict
import torch
from transformers import PreTrainedModel


class DistillationLossInput(NamedTuple):
    weight: float
    teacher_loss_input: torch.Tensor
    student_loss_input: torch.Tensor


class DistillationStrategy:
    """
    Distillation strategies create tensors from the teacher and student models which are inputs to the loss function.

    Distillation loss can be calculated based on any number of model features, including
    - logits
    - specific activations (`hidden_states`)
    - attention scones (`attentions`)

    A DistillationStrategy doesn't apply loss, but it defines the input pairs and weights from which loss is calculated.

    Caveat: Teacher and student models can have differing configurations, which is handled by a subset of strategies.
            E.g. if a student has fewer layers than a teacher, you cannot strictly train on activation pairs.
    """
    required_equivalent_config: List[str]
    forward_pass_kwargs: Dict

    def __init__(self, teacher_config, student_config):
        self._assert_required_configurations_equivalent(teacher_config, student_config)
        self.teacher_config = teacher_config
        self.student_config = student_config

    def features_to_loss_inputs(
            self,
            teacher_output: torch.Tensor,
            student_output: torch.Tensor,
    ) -> List[DistillationLossInput]:
        """Convert features generated by `get_loss_inputs` into distillation loss inputs"""
        ...

    def get_loss_inputs(
            self,
            teacher_model: PreTrainedModel,
            student_model: PreTrainedModel,
            inputs
    ) -> List[DistillationLossInput]:
        """
        Apply teacher_model and student_model forward pass to inputs.
        Determine pairs from which loss will be determined and weights.
        """
        with torch.no_grad():
            teacher_features = teacher_model(**inputs, **self.forward_pass_kwargs)
        student_features = student_model(**inputs, **self.forward_pass_kwargs)

        return self.features_to_loss_inputs(teacher_features, student_features)

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


class LogitsStrategy(DistillationStrategy):
    """
    Strategies Loss Inputs:
    - logits of forward pass
    """
    required_equivalent_config = ["vocab_size"]
    forward_pass_kwargs = {}

    @staticmethod
    def features_to_loss_inputs(teacher_output, student_output):
        loss_input = DistillationLossInput(
            weight=1,
            teacher_loss_input=teacher_output.logits,
            student_loss_input=student_output.logits
        )
        return [loss_input]


class ActivationsStrategy(DistillationStrategy):
    """
    Strategies Loss Inputs:
    - all layers activations

    Only use this strategy if output embeddings are identical to teacher.
    """
    required_equivalent_config = ["hidden_size", "num_hidden_layers"]
    forward_pass_kwargs = {"output_hidden_states": True}

    @staticmethod
    def features_to_loss_inputs(teacher_output, student_output):
        return [
            DistillationLossInput(
                weight=1 / len(student_output.hidden_states),
                teacher_loss_input=teacher_hidden,
                student_loss_input=student_hidden
            )
            for teacher_hidden, student_hidden
            in zip(teacher_output.hidden_states, student_output.hidden_states)
        ]


class AttentionsStrategy(DistillationStrategy):
    """
    Strategies Loss Inputs:
    - all attention blocks attention scores

    Only use this strategy if output embeddings are identical to teacher.
    """
    required_equivalent_config = ["hidden_size", "num_hidden_layers"]
    forward_pass_kwargs = {"output_attentions": True}

    @staticmethod
    def features_to_loss_inputs(teacher_output, student_output):
        return [
            DistillationLossInput(
                weight=1,
                teacher_loss_input=teacher_hidden,
                student_loss_input=student_hidden
            )
            for teacher_hidden, student_hidden
            in zip(teacher_output.attentions, student_output.attentions)
        ]


class LogitsActivationsStrategy(DistillationStrategy):
    """
    Strategies Loss Inputs:
    - logits
    - all layers activations
    """
    required_equivalent_config = ["vocab_size", "hidden_size", "num_hidden_layers"]
    forward_pass_kwargs = {"output_hidden_states": True}

    @staticmethod
    def features_to_loss_inputs(teacher_output, student_output):
        activation_inputs = ActivationsStrategy.features_to_loss_inputs(teacher_output, student_output)
        logit_input = LogitsStrategy.features_to_loss_inputs(teacher_output, student_output)
        logit_input.weight = len(activation_inputs)
        return activation_inputs + [logit_input]


class LogitsActivationsAttentionsStrategy(DistillationStrategy):
    """
    Strategies Loss Inputs:
    - logits
    - all layers activations
    - all attention blocks attention scores
    """
    required_equivalent_config = ["vocab_size", "hidden_size", "num_hidden_layers"]
    forward_pass_kwargs = {"output_hidden_states": True, "output_attentions": True}

    @staticmethod
    def features_to_loss_inputs(teacher_output, student_output):
        return (
            LogitsStrategy.features_to_loss_inputs(teacher_output, student_output) +
            ActivationsStrategy.features_to_loss_inputs(teacher_output, student_output) +
            AttentionsStrategy.features_to_loss_inputs(teacher_output, student_output)
        )


# TODO: activation pairs


STRATEGIES = {
    "logits": LogitsStrategy,
    "activations": ActivationsStrategy,
    "attentions": ActivationsStrategy,
    "logits_activations": LogitsActivationsStrategy,
    "logits_activations_attentions": LogitsActivationsAttentionsStrategy,
}
