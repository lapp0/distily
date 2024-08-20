from transformers import TrainingArguments, HfArgumentParser
from dataclasses import dataclass, field
import typing


def StrBoolTupleType(arg_str: str) -> typing.Tuple[str, bool]:
    if "," in arg_str:
        s, b = arg_str.split(",")
        return str(s), (b.lower() in ("true", "1"))
    else:
        return arg_str, False


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
    reinitialize_weights: typing.Optional[str] = None  # TODO: field
    copy_teacher_modules: typing.Optional[typing.List[StrBoolTupleType]] = field(
        default_factory=lambda: [("lm_head", False)],
        metadata={"help": (
            "List of tuples with module name and is_frozen boolean to copy modules from teacher to student. "
            "Default: copy the LM head, and make it trainable"
        )}
    )
    student_model_as_bitnet: bool = field(
        default=False,
        metadata={"help": "Make student model a bitnet model."}
    )

    # TODO: Full field
    # TODO: validator, require pytorch 2.5.0 for compile
    student_model_compile: bool = False

    dropout: typing.Optional[float] = None


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
    teacher_model_compile: bool = False


@dataclass
class DatasetArguments:
    dataset_uri: str = "wikimedia/wikipedia"
    dataset_subset: str = "20231101.en"
    dataset_split: str = "train"
    dataset_column_name: str = "text"
    dataset_sample_size: int = 250000
    dataset_test_size: float = 0.01


@dataclass
class DistillationObjectiveArguments:
    # TODO: make fields, clean up
    logits_weight: float = 1
    logits_loss_fn: str = "kl"

    hs_weight: float = 0
    hs_loss_fn: typing.Optional[str] = None
    hs_layer_mapper: typing.Optional[str] = None
    hs_projector: typing.Optional[str] = None

    attn_weight: float = 0
    attn_loss_fn: typing.Optional[str] = None
    attn_layer_mapper: typing.Optional[str] = None
    attn_projector: typing.Optional[str] = None


@dataclass
class DistillationTrainingArguments(TrainingArguments):

    ##################################
    # Distillation Training parameters
    ##################################
    train_embeddings: bool = field(
        default=True,
        metadata={"help": "If True, trains new embeddings from scratch. Else, use teachers input / output embeddings"}
    )

    # TODO: add extra metric evaluators

    # extra helper specific to this trainer
    eval_teacher_metrics: bool = True  # TODO: use field
    eval_on_end: bool = True

    #####################################################
    # TrainingArguments parameters with sane defaults set
    #####################################################

    # optimize convergence to final model
    learning_rate: float = 4e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.2
    lr_scheduler_type: str = "constant"
    num_train_epochs: float = 1.0
    optim: str = "paged_lion_32bit"

    # smaller batch sizes perform better
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1

    # optimize performance and memory
    per_device_eval_batch_size: int = 8  # TODO: auto-find?
    gradient_checkpointing: bool = True

    # Fixes
    gradient_checkpointing_kwargs = {"use_reentrant": False}

    # logging / evaluation
    logging_steps: int = 25
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    eval_on_start: bool = True
    report_to: str = "tensorboard"


parser = HfArgumentParser((
    DistillationTrainingArguments,
    DistillationObjectiveArguments,
    StudentModelArguments,
    TeacherModelArguments,
    DatasetArguments
))


def get_args():
    return parser.parse_args_into_dataclasses()
