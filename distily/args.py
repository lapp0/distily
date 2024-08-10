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
    # TODO: Full field
    # TODO: validator, require pytorch 2.5.0 for compile
    student_model_compile: bool = True



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
    # TODO: Full field
    # TODO: validator, require pytorch 2.5.0 for compile
    teacher_model_compile: bool = True


@dataclass
class DatasetArguments:
    dataset_uri: str = "wikimedia/wikipedia"
    dataset_subset: str = "20231101.en"
    dataset_split: str = "train"
    dataset_column_name: str = "text"
    dataset_sample_size: int = 250000
    dataset_test_size: float = 0.01


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
    distillation_objective: str = field(
        default="legacy",
        metadata={"help": "DistillationObjective callable which calculate loss"}
    )  # TODO: document how to set Activation Loss Pairs
    train_embeddings: bool = field(
        default=True,
        metadata={"help": "If True, trains new embeddings from scratch. Else, use teachers input / output embeddings"}
    )

    # TODO: add extra metric evaluators

    # extra helper specific to this trainer
    eval_on_end: bool = True

    #####################################################
    # TrainingArguments parameters with sane defaults set
    #####################################################

    # optimize convergence to final model
    learning_rate: float = 4e-5
    max_grad_norm: float = 100.0
    lr_scheduler_type: str = "constant"
    num_train_epochs: float = 1.0

    # optimize performance and memory
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True

    # Fixes
    gradient_checkpointing_kwargs = {"use_reentrant": False}

    # logging / evaluation
    logging_steps: int = 1
    eval_strategy: str = "steps"
    eval_steps: int = 500
    eval_on_start: bool = True
    report_to: str = "tensorboard"

    # push to hub


parser = HfArgumentParser((
    DistillationTrainingArguments,
    StudentModelArguments,
    TeacherModelArguments,
    DatasetArguments
))


def get_args():
    return parser.parse_args_into_dataclasses()
