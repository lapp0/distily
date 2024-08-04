from typing import Union, List, Optional, Callable
from transformers import Trainer
import torch
from torch.nn import functional as F
import logging
import gc


def mse_loss(student_features, teacher_features):
    return F.mse_loss(student_features, teacher_features)


def kl_divergence_loss(student_features, teacher_features):
    student_log_prob = F.log_softmax(student_features, dim=-1)
    teacher_prob = F.softmax(teacher_features, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def reverse_kl_divergence_loss(student_features, teacher_features):
    teacher_log_prob = F.log_softmax(teacher_features, dim=-1)
    student_prob = F.softmax(student_features, dim=-1)
    return F.kl_div(teacher_log_prob, student_prob, reduction="batchmean")


def cakld_loss(student_features, teacher_features, beta_prob=0.5):
    teacher_output_log_prob = F.log_softmax(teacher_features, dim=-1)
    student_output_soft = F.softmax(student_features, dim=-1)
    reverse_kl = F.kl_div(teacher_output_log_prob, student_output_soft, reduction="none").sum(-1)

    student_output_log_prob = F.log_softmax(student_features, dim=-1)
    teacher_output_soft = F.softmax(teacher_features, dim=-1)
    forward_kl = F.kl_div(student_output_log_prob, teacher_output_soft, reduction="none").sum(-1)

    kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
    return kl_loss.mean()


def jsd_loss(student_features, teacher_features, beta_prob=0.5):
    student_prob = F.softmax(student_features, dim=-1)
    teacher_prob = F.softmax(teacher_features, dim=-1)

    c_prob = beta_prob * teacher_prob + (1 - beta_prob) * student_prob
    c_log_prob = c_prob.log()

    kl_loss_f = beta_prob * F.kl_div(c_log_prob, teacher_prob, reduction="none").sum(-1)
    kl_loss_r = (1 - beta_prob) * F.kl_div(c_log_prob, student_prob, reduction="none").sum(-1)

    kl_loss = kl_loss_f + kl_loss_r
    return kl_loss.mean()


class DistillationTrainer(Trainer):
    loss_fn_map = {
        "mse": mse_loss,
        "kl": kl_divergence_loss,
        "reverse_kl": reverse_kl_divergence_loss,
        "cakld": cakld_loss,
        "jsd": jsd_loss
    }

    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        loss_fn: Optional[str] = None,
        activation_loss_pairs: Union[None, List[int], bool] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, model=student_model, tokenizer=tokenizer, **kwargs)
        self.teacher_model = teacher_model

        loss_fn = loss_fn or "reverse_kl"
        if isinstance(loss_fn, str):
            try:
                self.loss_fn = self.loss_fn_map[loss_fn.lower()]
            except KeyError:
                raise ValueError(f"Unsupported loss function: {self.loss_fn}")
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
                    metrics[f"eval_{evaluator_name}"] = float(evaluator(self.model))
                    gc.collect_garbage()
                    torch.cuda.empty_cache()
            self.model.train()

            self.log(metrics)

        metrics.update(
            super().evaluate(*args, **kwargs)
        )

        return metrics

    def create_model_card(self, *args, tags=None, finetuned_from=None, **kwargs):
        # TODO: randomly initialized weights, distilled from teacher model `{self.teacher_model.config._name_or_path}`
        return super().create_model_card(
            *args,
            tags=(tags or []) + ["Distily"],
            **kwargs
        )

    def eval_and_log_teacher_metrics(self):
        """TODO: This doesn't work properly!"""
        base_model_results = {}
        with torch.no_grad():
            for evaluator_name, evaluator in self.args.extra_evaluators.items():
                base_model_results[f"eval_{evaluator_name}"] = evaluator(self.teacher_model)
        base_model_results["epoch"] = 0
        self.log(base_model_results)
