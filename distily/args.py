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
    # TODO:
    # clean
    eval_and_log_teacher_metrics: bool = False

    # TODO: add
    # Activation loss pairs
    # extra metric evaluators
    pass


def get_args():
    parser = HfArgumentParser((DistillationTrainingArguments, StudentModelArguments, TeacherModelArguments))
    return parser.parse_args_into_dataclasses()
