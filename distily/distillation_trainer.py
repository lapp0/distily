from typing import Callable, List, Dict
import collections
import logging
import os

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

### Model Results
{eval_table}

### Framework versions
{framework_versions}
"""


class DistillationTrainer(transformers.Trainer):
    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        *args,
        **kwargs
    ):
        super().__init__(*args, model=student_model, tokenizer=tokenizer, **kwargs)
        self.teacher_model = teacher_model

        # prepare loss_fn
        if isinstance(self.args.loss_fn, str):
            try:
                self.loss_fn = distily.distill_loss.LOSS_FUNCTIONS[self.args.loss_fn.lower()]
            except KeyError:
                raise ValueError(f"Unsupported loss function: {self.args.loss_fn}")
        elif isinstance(self.args.loss_fn, Callable):
            self.loss_fn = self.args.loss_fn
        else:
            raise TypeError(f"invalid loss_fn: `{self.args.loss_fn}`")

        # prepare distillation_strategy
        if isinstance(self.args.distillation_strategy, distily.distillation_strategy.DistillationStrategy):
            self.distillation_strategy = self.args.distillation_strategy
        elif (
                isinstance(self.args.distillation_strategy, type) and
                issubclass(self.args.distillation_strategy, distily.distillation_strategy.DistillationStrategy)
        ):
            self.distillation_strategy = self.args.distillation_strategy(teacher_model.config, student_model.config)
        elif isinstance(self.args.distillation_strategy, str):
            distillation_strategy_cls = distily.distillation_strategy.STRATEGIES[self.args.distillation_strategy]
            self.distillation_strategy = distillation_strategy_cls(teacher_model.config, student_model.config)
        else:
            raise TypeError(f"invalid distillation_strategy: `{self.args.distillation_strategy}`")

        self.log_trainer_details()

    def log_trainer_details(self):
        logging.info("Student model: `{TODO}`")
        # TODO:
        # student / teacher model names / shapes
        # logits (y/n)
        # activation pair transfers

    def compute_loss(self, model, inputs, return_outputs=False):
        all_loss_inputs = self.distillation_strategy.get_loss_inputs(self.teacher_model, model, inputs)
        loss = torch.sum(torch.stack([
            self.loss_fn(teacher_input, student_input) * weight / len(student_input)
            for weight, teacher_input, student_input in all_loss_inputs
        ]))
        loss /= sum([inp.weight for inp in all_loss_inputs])  # normalize so weights add to 1

        if return_outputs:
            # TODO: real output
            return loss, torch.tensor([1.0])
        return loss

    def evaluate(self, *args, metric_key_prefix="eval", **kwargs):
        metrics = {}
        if metric_key_prefix == "eval":
            self.model.eval()
            with torch.no_grad():
                for evaluator_name, evaluator in self.args.extra_evaluators.items():
                    metrics[f"eval_{evaluator_name}"] = float(evaluator(
                        self.model,
                        self.args.per_device_eval_batch_size
                    ))
            self.model.train()

            self.log(metrics)

        metrics.update(
            super().evaluate(*args, **kwargs)
        )

        return metrics

    def create_model_card(self, *args, tags=None, finetuned_from=None, **kwargs):
        super().create_model_card(
            *args,
            tags=(tags or []) + ["Distily"],
            **kwargs
        )

        step_evals = collections.defaultdict(dict)
        for log_line in self.state.log_history:
            extracted_logs = {
                k: transformers.modelcard._maybe_round(v)
                for k, v in log_line.items()
                if k.endswith('_loss') or k.startswith('eval_')
            }
            if extracted_logs:
                step_evals[log_line["step"]].update({
                    "epoch": log_line.get("epoch"),
                    **extracted_logs
                })
        eval_lines = [{"step": step, **value} for step, value in sorted(step_evals.items())]

        eval_results = eval_lines[-1]
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
            "distillation_strategy": str(self.args.distillation_strategy),
            "loss_fn": str(self.args.loss_fn),
            "train_embeddings": str(self.args.train_embeddings),
            **hyperparameters
        }

        # TODO: add strategy, loss_fn, other parameters and details

        # TODO: add
        # model_card.data["metrics"] = ...
        # model_card.data["datasets"] = ...

        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        model_card = ModelCard.load(model_card_filepath)
        model_card.data["library_name"] = "distily"

        model_card.text = MODEL_CARD_TEMPLATE.format(
            model_name=self.args.output_dir,
            teacher_model=self.teacher_model.config._name_or_path,
            dataset_name="(unspecified)",  # TODO
            eval_result="\n".join([
                f"- {name}: {transformers.modelcard._maybe_round(value)}"
                for name, value in eval_results.items()
            ]),
            hyperparameters="\n".join([f"- {name}: {value}" for name, value in hyperparameters.items()]),
            eval_table=self._to_markdown_table(eval_lines),
            framework_versions=framework_versions
        )
        with open(model_card_filepath, "w") as f:
            f.write(model_card)

    @staticmethod
    def _to_markdown_table(lines: List[Dict]) -> str:
        all_keys = sorted(set(key for row in lines for key in row))
        header = "| " + " | ".join(all_keys) + " |"
        separator = "| " + " | ".join("---" for _ in all_keys) + " |"
        rows = [
            "| " + " | ".join(str(row.get(key, "")) for key in all_keys) + " |"
            for row in lines
        ]
        return "\n".join([header, separator] + rows)

    def eval_and_log_teacher_metrics(self):
        """TODO: This doesn't work properly!"""
        base_model_results = {}
        with torch.no_grad():
            for evaluator_name, evaluator in self.args.extra_evaluators.items():
                base_model_results[f"eval_{evaluator_name}"] = float(evaluator(
                    self.teacher_model,
                    self.args.per_device_eval_batch_size
                ))
        base_model_results["epoch"] = 0
        self.log(base_model_results)
