from typing import Callable, List, Dict
import collections
import logging
import os
import gc

import transformers
import torch
from huggingface_hub import ModelCard

import distily


MODEL_CARD_TEMPLATE = """
# {model_name}

This student model is distilled from the teacher model [{teacher_model}](https://huggingface.co/{teacher_model}) using the dataset {dataset_name}.

The [Distily](https://github.com/lapp0/distily) library was used for this distillation.

It achieves the following results on the evaluation set:
{eval_result}

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed
-->

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
{hyperparameters}

### Resource Usage
Peak GPU Memory: {peakmem_gb} GB

### Eval-Phase Metrics
{eval_table}

### Framework versions
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
        *args,
        **kwargs
    ):
        super().__init__(*args, model=student_model, tokenizer=tokenizer, **kwargs)

        self.teacher_model = teacher_model
        self.distillation_objective = distillation_objective

        self.log_trainer_details()

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

        # TODO: include __version__ in distily
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

        # TODO: DistillationObjective needs a repr to include the loss fn, etc

        # TODO: add
        # model_card.data["metrics"] = ...
        # model_card.data["datasets"] = ...

        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        model_card = ModelCard.load(model_card_filepath)
        model_card.data["library_name"] = "Distily"

        model_card.text = MODEL_CARD_TEMPLATE.format(
            model_name=self.args.output_dir,
            teacher_model=self.teacher_model.config._name_or_path,
            dataset_name="(unspecified)",  # TODO
            eval_result="\n".join([
                f"- {name}: {transformers.modelcard._maybe_round(value)}"
                for name, value in eval_results.items()
            ]),
            hyperparameters="\n".join([f"- {name}: {value}" for name, value in hyperparameters.items()]),
            peakmem_gb=transformers.modelcard._maybe_round(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            eval_table=self._to_markdown_table(eval_lines),
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
