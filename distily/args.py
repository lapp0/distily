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
    student_use_liger_kernel: bool = True  # TODO: Field


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
class DatasetArguments:
    dataset_uri: str = "wikimedia/wikipedia"
    dataset_subset: str = "20231101.en"
    dataset_split: str = "train"
    dataset_column_name: str = "text"
    dataset_sample_size: int = 1_000_000
    dataset_max_seq_length: typing.Optional[None] = None
    dataset_test_size: float = 0.002
    dataset_shuffle: bool = True
    dataset_shuffle_seed: int = 42
    dataset_trust_remote_code: bool = False

@dataclass
class EvalArguments:
    grad_var_stats: bool = True

    ppl_evaluators: typing.List[typing.Dict] = field(
        default_factory=lambda: [
            dict(name="enwikippl", dataset="wikimedia/wikipedia", subset="20231101.en", split="train", sample_size=2500),
            dict(name="frwikippl", dataset="wikimedia/wikipedia", subset="20231101.fr", split="train", sample_size=1000),
            dict(name="zhwikippl", dataset="wikimedia/wikipedia", subset="20231101.zh", split="train", sample_size=1000),
        ],
        metadata={"help": "Default evaluation metrics with their parameters."}
    )
    ppl_extra_evaluators: typing.List[typing.Dict] = field(
        default_factory=list,
        metadata={"help": "Additional evaluation metrics to be used."}
    )
    harness_benchmarks: typing.List[typing.Dict] = field(
        default_factory=list,
        # official model release recommendation:
        # include lambda: ["wikitext", "boolq", "hellaswag", "glue", "ai2_arc", "mmlu", "math"]
        metadata={"help": "Benchmarks to compare student and teacher models at end of training."}
    )
    harness_benchmark_limit: int = field(
        default=5000,
        # official model release recommendation: set to None for official releases to measure all data points
        metadata={"help": "Limit the number of examples per task (only use this for testing), If <1, limit is %."}
    )
    harness_benchmark_bootstrap_iters: int = field(
        default=0,
        # official model release recommendation: set to None for official releases to measure error
        metadata={"help": "Number iter for bootstrap stats for stderr. Set to 0 to skip stderr calc. "}
    )

@dataclass
class DistillationObjectiveArguments:
    # TODO: make fields, clean up
    logits_weight: float = 1
    logits_loss_fn: str = "kl"

    hs_weight: float = 0
    hs_loss_fn: str = None
    hs_layer_mapper: str = None
    hs_norm: typing.Optional[str] = None
    hs_projector: typing.Optional[str] = None

    # TODO: enable once more when its params are fixed for attn matrices
    attn_weight: float = 0.0
    attn_loss_fn: str = "raw_mse"
    attn_layer_mapper: str = "layer-2"
    attn_norm: typing.Optional[str] = "layernorm_teacher_only_affine"
    attn_projector: typing.Optional[str] = "orthogonal"


@dataclass
class DistillationTrainingArguments(TrainingArguments):

    ##################################
    # Distillation Training parameters
    ##################################

    # extra helper specific to this trainer
    eval_teacher_metrics: bool = True  # TODO: use field

    #####################################################
    # TrainingArguments parameters with sane defaults set
    #####################################################

    # optimize convergence to final model
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "polynomial"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"lr_end": 2e-5, "power": 0.7})
    num_train_epochs: float = 1.0
    optim: str = "paged_lion_32bit"

    # larger batches appear to train better?
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # optimize performance and memory
    per_device_eval_batch_size: int = 4  # TODO: auto-find?
    gradient_checkpointing: bool = True
    bf16: bool = True

    # TODO: enable torch compile when this incompatibility with use_liger_kernel is fixed
    # -----------------------
    # /opt/conda/lib/python3.10/site-packages/torch/autograd/graph.py:825:
    # UserWarning: cuDNN SDPA backward got grad_output.strides() != output.strides(),
    # attempting to materialize a grad_output with matching strides...
    # (Triggered internally at ../aten/src/ATen/native/cudnn/MHA.cpp:674.)
    # torch_compile: bool = True  # TODO: Field

    # Fixes
    gradient_checkpointing_kwargs = {"use_reentrant": False}

    # logging / evaluation
    logging_steps: int = 16
    save_steps: int = 5000
    save_total_limit = 4
    eval_strategy: str = "steps"
    eval_steps: int = 5000
    eval_on_start: bool = True
    eval_on_end: bool = True
    report_to: str = "tensorboard"


parser = HfArgumentParser((
    DistillationTrainingArguments,
    DistillationObjectiveArguments,
    StudentModelArguments,
    TeacherModelArguments,
    DatasetArguments,
    EvalArguments
))


def get_args():
    return parser.parse_args_into_dataclasses()
