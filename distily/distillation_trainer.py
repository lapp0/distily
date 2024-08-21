from typing import Callable, List, Dict
from dataclasses import asdict
import collections
import logging
import os
import gc
import sys

import transformers
import torch
from huggingface_hub import ModelCard

import distily


MODEL_CARD_TEMPLATE = """
# Summary

- Model Name: `{model_name}`
- Distilled using the [Distily](https://github.com/lapp0/distily) library
- Teacher Model: [{teacher_model}](https://huggingface.co/{teacher_model})
- Train Dataset: [{dataset_name}](link_to_dataset)

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment.

# Model description

More information needed

# Intended uses & limitations

More information needed
-->

# Student Model (`{model_name}`) Architecture:
- **Architecture**: {student_model_architecture}
- **Total Parameters**: {student_total_params}
- **Data Type (dtype)**: {student_model_dtype}
  - **Quantization:** {student_quantization}
- **Model Size**: {student_model_size}

<details>
  <summary>Student Model Architecture Details</summary>
  {model_repr}
</details>

# Teacher Model Architecture:
- **Architecture**: {teacher_model_architecture}
- **Total Parameters**: {teacher_total_params}
- **Data Type (dtype)**: {teacher_model_dtype}
  - Quantization: {teacher_quantization}
- **Model Size**: {teacher_model_size}

<details>
  <summary>Teacher Model Architecture Details</summary>
  {teacher_model_repr}
</details>

# Architecture Diff:

<details>
<summary>Expand</summary>
```diff
{model_diff_repr}
```
</details>


# Evaluation Metrics Comparison

{eval_table}

# Resource Usage Comparison

{resource_table}


# Train Dataset
Trained on {token_count} tokens from the [{dataset_name}](link_to_dataset) dataset.

- Subset / Split: [subset={dataset_subset_name} split={dataset_split_name}](link_to_dataset)
- Train Samples: {num_train_samples}

# Training Loss Function
**Logits**:
{logit_details}

**Hidden States**:
{hs_loss_details}

**Attention**:
{attn_loss_details}


# Hyperparameters
The following hyperparameters were used during training:
{hyperparameters}


# Framework Versions
{framework_versions}
"""

# TODO: add 'train_runtime', 'train_samples_per_second', 'train_steps_per_second' hardware info


class DistillationTrainer(transformers.Trainer):
    def __init__(
        self,
        distillation_objective,
        student_model,
        teacher_model,
        tokenizer,
        all_args=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, model=student_model, tokenizer=tokenizer, **kwargs)

        self.all_args = all_args or {}

        self.teacher_model = teacher_model
        self.distillation_objective = distillation_objective

        self.log_trainer_details()

,    @classmethod
    def from_args(
            cls,
            training_args,
            distillation_objective_args,
            student_model_args,
            teacher_model_args,
            dataset_args
    ):

        teacher_model, tokenizer = distily.models.get_teacher_model_tokenizer(teacher_model_args)
        student_model = distily.models.get_student_model(student_model_args, teacher_model)

        # TODO: don't hardcode max length
        max_seq_len = 1024
        # TODO: don't hardcode this
        training_args.extra_evaluators = distily.metrics.get_all_metric_evaluators(tokenizer)

        train_dataset, test_dataset = distily.data.get_dataset(dataset_args, tokenizer, max_seq_len)

        distillation_objective = distily.objectives.DistillationObjective(**asdict(distillation_objective_args))

        return cls(
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            distillation_objective=distillation_objective,
            all_args=dict(
                distillation_objective_args=distillation_objective_args,
                student_model_args=student_model_args,
                teacher_model_args=teacher_model_args,
                dataset_args=dataset_args,
            )
        )

    def from_kwargs(cls, **kwargs):
        parsed_args_tuple = distily.args.parser.parse_dict(
            kwargs,
            allow_extra_keys=True
        )
        return cls.from_args(*parsed_args_tuple)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.args.eval_on_end:
            self.evaluate()

    def log_trainer_details(self):
        logging.info("Student model: `{TODO}`")
        # TODO:
        # student / teacher model names / shapes
        # logits (y/n)
        # activation pair transfers

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_dict = self.distillation_objective(self.teacher_model, model, inputs)
        loss = loss_dict.pop("loss")

        # if train step, log metrics
        if not return_outputs:
            self.log({
                "step": self.state.global_step,
                **{k: float(v) for k, v in loss_dict.items()},
            })

        if return_outputs:
            # TODO: real output, this is nothing of use
            return loss, torch.tensor([1.0])
        return loss

    def evaluate(self, *args, metric_key_prefix="eval", **kwargs):
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        metrics = {}
        if metric_key_prefix == "eval":
            with torch.no_grad():
                for evaluator_name, evaluator in self.args.extra_evaluators.items():
                    metrics[f"eval_{evaluator_name}"] = float(evaluator(
                        self.model,
                        self.args.per_device_eval_batch_size
                    ))

            self.log(metrics)

        metrics.update(
            super().evaluate(*args, **kwargs)
        )

        self.model.train()
        return metrics

    def create_model_card(self, *args, **kwargs):
        super().create_model_card(*args, **kwargs)

        # Student model details
        student_model_architecture = self.model.config.architectures[0] if hasattr(self.model.config, 'architectures') else "Unknown"
        student_total_params = sum(p.numel() for p in self.model.parameters())
        student_model_dtype = next(self.model.parameters()).dtype
        student_quantization = "Not Quantized"  # Set based on the quantization details, if any
        student_model_size = sys.getsizeof(self.model.state_dict().keys()) / (1024 ** 2)

        # Teacher model details
        teacher_model_architecture = self.teacher_model.config.architectures[0] if hasattr(self.teacher_model.config, 'architectures') else "Unknown"
        teacher_total_params = sum(p.numel() for p in self.teacher_model.parameters())
        teacher_model_dtype = next(self.teacher_model.parameters()).dtype
        teacher_quantization = "Not Quantized"  # Set based on the quantization details, if any
        teacher_model_size = sys.getsizeof(self.teacher_model.state_dict().keys()) / (1024 ** 2)

        step_evals = collections.defaultdict(dict)
        for log_line in self.state.log_history:
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

        if self.args.eval_teacher_metrics:
            step_evals["**teacher eval**"] = {
                k: transformers.modelcard._maybe_round(v)
                for k, v in self.eval_teacher_metrics().items()
            }

        eval_lines = [{"step": step, **value} for step, value in sorted(step_evals.items())]

        eval_results = dict(eval_lines[-1])
        eval_results.pop("step", None)
        eval_results.pop("epoch", None)

        import pkg_resources, datasets
        framework_versions = "\n".join([
            f"- Distily {pkg_resources.get_distribution('distily').version}",
            f"- Transformers {transformers.__version__}",
            f"- Pytorch {torch.__version__}",
            f"- Datasets {datasets.__version__}",
        ])

        hyperparameters = transformers.modelcard.extract_hyperparameters_from_trainer(self)
        hyperparameters = {
            "distillation_objective": repr(self.distillation_objective),
            "train_embeddings": str(self.args.train_embeddings),
            **hyperparameters
        }

        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        model_card = ModelCard.load(model_card_filepath)
        model_card.data["library_name"] = "Distily"

        if self.all_args.get("train_dataset"):
            model_card.data["datasets"] = [self.all_args["train_dataset"].dataset_uri]
            dataset_kwargs = dict(
                dataset_name=self.all_args["train_dataset"].dataset_uri,
                token_count=self.all_args["train_dataset"].dataset_sample_size,
                dataset_subset_name=self.all_args["train_dataset"].dataset_subset,
                dataset_split_name=self.all_args["train_dataset"].dataset_split,
            )
        else:
            dataset_kwargs = dict(
                dataset_name="unspecified",
                dataset_subset_name="unspecified",
                dataset_split_name="unspecified",
            )

        token_count = self.all_args["train_dataset"].dataset_sample_size,

        import pdb;pdb.set_trace()

        model_card.text = MODEL_CARD_TEMPLATE.format(
            model_name=self.args.output_dir,
            teacher_model=self.teacher_model.config._name_or_path,
            student_model_architecture=student_model_architecture,
            student_total_params=f"{student_total_params:,}",
            student_model_dtype=str(student_model_dtype),
            student_quantization=student_quantization,
            student_model_size=f"{student_model_size:.2f} MB",
            model_repr=repr(self.model),
            teacher_model_repr=repr(self.teacher_model),
            teacher_model_architecture=teacher_model_architecture,
            teacher_total_params=f"{teacher_total_params:,}",
            teacher_model_dtype=str(teacher_model_dtype),
            teacher_quantization=teacher_quantization,
            teacher_model_size=f"{teacher_model_size:.2f} MB",
            model_diff_repr="",  # Needs custom implementation if required
            eval_table=self._to_markdown_table(eval_lines),
            resource_table="",  # Implement based on your resource metrics
            num_train_samples=len(self.train_dataset),
            logit_details="",  # Extracted from distillation objective or logs
            hs_loss_details="",  # Extracted from distillation objective or logs
            attn_loss_details="",  # Extracted from distillation objective or logs
            hyperparameters="\n".join([f"- {name}: {value}" for name, value in hyperparameters.items()]),
            framework_versions=framework_versions
        )
        model_card.save(model_card_filepath)

    @staticmethod
    def _to_markdown_table(lines: List[Dict]) -> str:
        all_keys = sorted(
            set(key for row in lines for key in row),
            key=lambda s: (s != "step", s != "epoch", s)
        )
        header = "| " + " | ".join([k.removeprefix("eval_") for k in all_keys]) + " |"
        separator = "| " + " | ".join("---" for _ in all_keys) + " |"
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

    def eval_teacher_metrics(self):
        teacher_model_results = {}
        with torch.no_grad():
            for evaluator_name, evaluator in self.args.extra_evaluators.items():
                teacher_model_results[f"eval_{evaluator_name}"] = float(evaluator(
                    self.teacher_model,
                    self.args.per_device_eval_batch_size
                ))
        return teacher_model_results
