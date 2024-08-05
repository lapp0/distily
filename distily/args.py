from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field
import typing


@dataclass
class StudentModelArguments:
    student_model_name_or_path: typing.Optional[str] = field(
        default=None,
        metadata={"help": "Student model URI or path to finetune. If unset, student is randomly initialized."}
    )
    student_config_name_or_path: typing.Optional[str] = field(
        default=None,
        metadata={"help": "Student config URI. If unset, student config is derived from teacher config."}
    )
    student_model_config: typing.Optional[dict] = field(
        default=None,
        metadata={"help": "Config dict of student model. Unset parameters default to set models config."}
    )
    student_model_as_bitnet: bool = field(
        default=False,
        metadata={"help": "Make student model a bitnet model."}
    )


@dataclass
class TeacherModelArguments:
    teacher_model_name_or_path: str = field(
        metadata={"help": "Teacher model URI or path."}
    )
    teacher_load_in_8bit: typing.Optional[bool] = field(
        default=False,
        metadata={"help": "load the teacher model in 8 bits precision"}
    )
    teacher_load_in_4bit: typing.Optional[bool] = field(
        default=False,
        metadata={"help": "Load the teacher model in 4 bits precision"}
    )


@dataclass
class DistillationTrainingArguments(TrainingArguments):

    ##################################
    # Distillation Training parameters
    ##################################
    eval_teacher_metrics: bool = True  # TODO: use field
    loss_fn: typing.Union[str] = field(
        default="reverse_kl",
        metadata={"help": "Loss function for distillation"}
    )
    distillation_strategy: str = field(
        default="logits_activations",
        metadata={"help": "Strategy determining which forward-pass features to incorporate into loss function."}
    )  # TODO: document how to set Activation Loss Pairs
    train_embeddings: bool = field(
        default=True,
        metadata={"help": "If True, trains new embeddings from scratch. Else, use teachers input / output embeddings"}
    )

    # TODO: add extra metric evaluators

    #####################################################
    # TrainingArguments parameters with sane defaults set
    #####################################################

    # optimize convergence to final model
    learning_rate: float = 1e-4
    max_grad_norm: float = 100.0
    lr_scheduler_type: str = "cosine"
    num_train_epochs: float = 1.0

    # optimize performance and memory
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    optim: str = "paged_adamw_32bit"
    gradient_checkpointing: bool = True

    # logging / evaluation
    logging_steps: int = 16
    eval_strategy: str = "steps"
    eval_steps: int = 2000
    eval_on_start: bool = True


def get_args():
    parser = HfArgumentParser((DistillationTrainingArguments, StudentModelArguments, TeacherModelArguments))
    return parser.parse_args_into_dataclasses()
