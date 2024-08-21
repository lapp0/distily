from dataclasses import asdict
import collections
import difflib
import torch
import transformers
import typing
import distily
import datasets


MODEL_CARD_TEMPLATE = """

# Summary

Distilled with [Distily](https://github.com/lapp0/distily) library
using teacher model [{teacher_model}](https://huggingface.co/{teacher_model})
on dataset [{dataset_name}](https://huggingface.co/datasets/{dataset_name}).

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment.

# Model description

More information needed

# Intended uses & limitations

More information needed
-->

# Model Architecture:
- **Architecture**: `{student_model_architecture}`
- **Total Parameters**: {student_total_params}
- **Data Type (dtype)**: {student_model_dtype}
- **Model Size**: {student_model_size}


# Evaluation Metrics Comparison

{eval_table}

# Resource Usage Comparison

{resource_table}

# Distillation (Teacher -> Student) Architecture Difference:

- **Architecture**: `{teacher_model_architecture}` -> `{student_model_architecture}`
- **Total Parameters**: {teacher_total_params} -> {student_total_params}
- **Data Type (dtype)**: {teacher_model_dtype} -> {student_model_dtype}
- **Model Size**: {teacher_model_size} -> {student_model_size}

<details>
<summary>Module Diff Details</summary>

```diff
{model_diff_repr}
```

</details>
<br/>

# Train Dataset
Trained on {token_count:,} tokens from the [{dataset_name}](https://huggingface.co/datasets/{dataset_name}) dataset.

- Num Samples: `{num_train_samples:,}`
- Subset: `{dataset_subset_name}`
- Split: `{dataset_split_name}`


# Training Objective

```
{objective_details}
```

# Hyperparameters
The following hyperparameters were used during training:

<details>
<summary>Expand</summary>

{hyperparameters}

</details>
<br/>


# Framework Versions
{framework_versions}
"""


def _to_markdown_table(lines: typing.List[typing.Dict]) -> str:
    all_keys = sorted(
        set(key for row in lines for key in row),
        key=lambda s: (s != "step", s != "epoch", s)
    )
    header = "| " + " | ".join([k.removeprefix("eval_") for k in all_keys]) + " |"
    separator = "| " + " | ".join(":---:" for _ in all_keys) + " |"
    sorted_lines = sorted(
        lines,
        key=lambda line: (
            line["step"].isnumeric(),
            float(line["step"]) if line["step"].isnumeric() else line["step"]
        )
    )
    rows = [
        "| " + " | ".join(str(row.get(key, "")) for key in all_keys) + " |"
        for row in sorted_lines
    ]
    return "\n".join([header, separator] + rows)


def create_model_card_text(trainer):
    # Student model details
    student_model_architecture = (
        trainer.model.config.architectures[0]
        if hasattr(trainer.model.config, 'architectures')
        else "Unknown"
    )
    student_total_params = trainer.model.num_parameters()
    student_model_dtype = next(trainer.model.parameters()).dtype
    student_model_size = trainer.model.get_memory_footprint() / (1024 ** 3)

    # Teacher model details
    teacher_model_architecture = (
        trainer.teacher_model.config.architectures[0]
        if hasattr(trainer.teacher_model.config, 'architectures')
        else "Unknown"
    )
    teacher_total_params = sum(p.numel() for p in trainer.teacher_model.parameters())
    teacher_model_dtype = next(trainer.model.parameters()).dtype
    teacher_model_size = trainer.teacher_model.get_memory_footprint() / (1024 ** 3)

    step_evals = collections.defaultdict(dict)
    for log_line in trainer.state.log_history:
        extracted_logs = {
            k: transformers.modelcard._maybe_round(v)
            for k, v in log_line.items()
            if k.startswith('eval_')
        }
        if extracted_logs:
            step_evals[transformers.modelcard._maybe_round(log_line["step"])].update({
                "epoch": transformers.modelcard._maybe_round(log_line.get("epoch")),
                **extracted_logs
            })

    if trainer.args.eval_teacher_metrics:
        step_evals["**teacher eval**"] = {
            k: transformers.modelcard._maybe_round(v)
            for k, v in trainer.eval_teacher_metrics().items()
        }

    eval_lines = [{"step": step, **value} for step, value in sorted(step_evals.items())]

    eval_results = dict(eval_lines[-1])
    eval_results.pop("step", None)
    eval_results.pop("epoch", None)

    framework_versions = "\n".join([
        f"- Distily {distily.__version__}",
        f"- Transformers {transformers.__version__}",
        f"- Pytorch {torch.__version__}",
        f"- Datasets {datasets.__version__}",
    ])

    included_training_args = [
        "gradient_accumulation_steps", "weight_decay", "max_grad_norm",
        "warmup_ratio", "warmup_steps", "gradient_checkpointing",
    ]
    additional_training_args = {k: v for k, v in asdict(trainer.args).items() if k in included_training_args}

    # includes lr, train_batch_size, eval_batch_size, seed,
    # optimizer, lr_scheduler_type / warmup_ratio, num_epochs
    hyperparameters = transformers.modelcard.extract_hyperparameters_from_trainer(trainer)
    hyperparameters.update({
        "distillation_objective": repr(trainer.distillation_objective),
        "train_embeddings": str(trainer.args.train_embeddings),
        "lr_scheduler": trainer.lr_scheduler,
        **asdict(trainer.all_args.get("student_model_args", {})),
        **asdict(trainer.all_args.get("teacher_model_args", {})),
        **asdict(trainer.all_args.get("dataset_args", {})),
        **additional_training_args,
    })

    if trainer.all_args.get("dataset_args"):
        dataset_kwargs = dict(
            dataset_name=trainer.all_args["dataset_args"].dataset_uri,
            dataset_subset_name=trainer.all_args["dataset_args"].dataset_subset,
            dataset_split_name=trainer.all_args["dataset_args"].dataset_split,
        )
    else:
        dataset_kwargs = dict(
            dataset_name="unspecified",
            dataset_subset_name="unspecified",
            dataset_split_name="unspecified",
        )

    model_diff_repr = "".join(
        difflib.unified_diff(
            repr(trainer.teacher_model).splitlines(keepends=True),
            repr(trainer.model).splitlines(keepends=True),
            fromfile="teacher model modules",
            tofile="student model modules"
        )
    )

    # TODO: Expand on this
    # - Hardware (GPU / total VRAM / CPU / total memory)
    # - Eval Performance of both models in terms of memory and speed
    # - 'train_runtime', 'train_samples_per_second', 'train_steps_per_second' hardware info
    resource_table = (
        "- VRAM Use: " +
        transformers.modelcard._maybe_round(torch.cuda.max_memory_allocated() / (1024 ** 3)) +
        " GB"
    )

    return MODEL_CARD_TEMPLATE.format(
        model_name=trainer.args.output_dir,
        teacher_model=trainer.teacher_model.config._name_or_path,
        student_model_architecture=student_model_architecture,
        student_total_params=f"{student_total_params:,}",
        student_model_dtype=str(student_model_dtype),
        student_model_size=f"{student_model_size:.2f} GB",
        teacher_model_architecture=teacher_model_architecture,
        teacher_total_params=f"{teacher_total_params:,}",
        teacher_model_dtype=str(teacher_model_dtype),
        teacher_model_size=f"{teacher_model_size:.2f} GB",
        model_diff_repr=model_diff_repr,
        eval_table=_to_markdown_table(eval_lines),
        resource_table=resource_table,
        num_train_samples=len(trainer.train_dataset),
        objective_details=trainer.distillation_objective,
        hyperparameters="\n".join([f"- {name}: `{value}`" for name, value in hyperparameters.items()]),
        framework_versions=framework_versions,
        token_count=sum(map(sum, trainer.train_dataset["attention_mask"])),
        **dataset_kwargs
    )
