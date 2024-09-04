from dataclasses import asdict
import shelve
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

<details>
<summary>Student Model Details</summary>

```
{student_model_repr}
```

</details>
<br/>

{"# Benchmark Metrics Comparison" if benchmark_table else ""}

{benchmark_table}

# Resource Usage

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


def _flatten_harness_results(model_results):
    return {
        f"{key} ({metric_key.split(',')[0]})": float(value)
        for key, metrics in model_results.items()
        for metric_key, value in metrics.items()
        if isinstance(value, (int, float))  # ignore aliases
    }


def _to_markdown_table(data: typing.Dict[str, typing.Dict], sigfig=4) -> str:

    def format_value(value):
        return f"{round(value, sigfig - len(str(int(value))))}" if isinstance(value, (int, float)) else value

    columns = sorted(data)
    metrics = sorted({m for v in data.values() for m in v})
    header = f"| Metric | {' | '.join(columns)} |"
    separator = "| :--- |" + " :--- |" * len(columns)
    rows = [
        f"| {m} | " + " | ".join(format_value(data[c].get(m, "")) for c in columns) + " |"
        for m in metrics
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

    eval_results = dict()
    for log_line in trainer.state.log_history:
        eval_results.update(log_line)

    eval_results.pop("step", None)
    eval_results.pop("epoch", None)

    with shelve.open(trainer.benchmarks_shelf) as db:
        benchmark_results = {
            model_label: _flatten_harness_results(db[model_label]["results"])
            for model_label in db.keys()
        }
        benchmark_table = _to_markdown_table(benchmark_results) if benchmark_results else None

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
        student_model_repr=repr(trainer.student_model),
        teacher_model_architecture=teacher_model_architecture,
        teacher_total_params=f"{teacher_total_params:,}",
        teacher_model_dtype=str(teacher_model_dtype),
        teacher_model_size=f"{teacher_model_size:.2f} GB",
        model_diff_repr=model_diff_repr,
        benchmark_table=benchmark_table,
        resource_table=resource_table,
        num_train_samples=len(trainer.train_dataset),
        objective_details=trainer.distillation_objective,
        hyperparameters="\n".join([f"- {name}: `{value}`" for name, value in hyperparameters.items()]),
        framework_versions=framework_versions,
        token_count=sum(map(sum, trainer.train_dataset["attention_mask"])),
        **dataset_kwargs
    )


def update_model_card(model_card, trainer):
    model_card.data["library_name"] = "Distily"
    if trainer.all_args.get("dataset_args"):
        model_card.data["datasets"] = [trainer.all_args["dataset_args"].dataset_uri]

    model_card.data["license"] = "creativeml-openrail-m"
    model_card.data["base_model"] = trainer.teacher_model.config._name_or_path,
    model_card.data["tags"] += "Distily"
    model_card.data["base_model_relation"] = "finetune"  # TODO: update to "distillation"

    model_card.text = create_model_card_text(trainer)

    return model_card
