from typing import Union, List, Optional, Callable
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
    loss_fn_map = {
        "mse": distily.distill_loss.mse_loss,
        "kl": distily.distill_loss.kl_divergence_loss,
        "reverse_kl": distily.distill_loss.reverse_kl_divergence_loss,
        "cakld": distily.distill_loss.cakld_loss,
        "jsd": distily.distill_loss.jsd_loss
    }

    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        activation_loss_pairs: Union[None, List[int], bool] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, model=student_model, tokenizer=tokenizer, **kwargs)
        self.teacher_model = teacher_model

        loss_fn = self.args.loss_fn or "reverse_kl"
        if isinstance(self.args.loss_fn, str):
            try:
                self.loss_fn = self.loss_fn_map[self.args.loss_fn.lower()]
            except KeyError:
                raise ValueError(f"Unsupported loss function: {self.args.loss_fn}")
        elif isinstance(loss_fn, Callable):
            self.loss_fn = loss_fn
        else:
            raise TypeError(f"invalid loss_fn: `{loss_fn}`")

        if activation_loss_pairs is None or activation_loss_pairs is False:
            self.activation_loss_pairs = []
        elif activation_loss_pairs is True:
            assert student_model.config.num_hidden_layers == teacher_model.config.num_hidden_layers  # TODO: explicit error
            self.activation_loss_pairs = [(i, i) for i in range(student_model.config.num_hidden_layers)]
        else:
            self.activation_loss_pairs = activation_loss_pairs

        self.log_trainer_details()

        # TODO: fix, hardcoded for model card generation purposes
        self.dataset


    def log_trainer_details(self):
        logging.info("Student model: `{TODO}`")
        # TODO:
        # student / teacher model names / shapes
        # logits (y/n)
        # activation pair transfers

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.activation_loss_pairs:
            with torch.no_grad():
                teacher_output = self.teacher_model(**inputs, output_hidden_states=True)
            student_output = model(**inputs, output_hidden_states=True)

            logit_loss = self.loss_fn(student_output.logits, teacher_output.logits)

            activation_losses = []
            for teacher_layer_idx, student_layer_idx in self.activation_loss_pairs:
                teacher_hidden_state = teacher_output.hidden_states[teacher_layer_idx]
                student_hidden_state = student_output.hidden_states[student_layer_idx]
                activation_losses.append(
                    torch.mean(self.loss_fn(teacher_hidden_state, student_hidden_state))
                )

            loss = torch.mean(torch.stack(activation_losses)) + torch.mean(logit_loss)

        else:
            with torch.no_grad():
                teacher_features = self.teacher_model(**inputs).logits
            student_features = model(**inputs).logits
            loss = torch.mean(
                self.loss_fn(student_features, teacher_features)
            )

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

        step_evals = {}
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
            hyperparameters="\n".join([f"- {name}: {value}" for name, value in self.hyperparameters.items()]),
            eval_table=transformers.modelcard.make_markdown_table(eval_lines),
            framework_versions=framework_versions
        )
        with open(model_card_filepath, "w") as f:
            f.write(model_card)

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
