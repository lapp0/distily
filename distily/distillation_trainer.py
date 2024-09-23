from dataclasses import asdict
import statistics
import collections
import os
import gc
import shelve

import transformers
import torch
from huggingface_hub import ModelCard

import distily


class DistillationTrainer(transformers.Trainer):
    def __init__(
        self,
        distillation_objective,
        student_model,
        teacher_model,
        tokenizer,
        evaluators,
        all_args=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, model=student_model, tokenizer=tokenizer, **kwargs)

        self.all_args = all_args or {}

        self.teacher_model = teacher_model
        self.distillation_objective = distillation_objective

        self.evaluators = evaluators

        self._extra_stats = []
        self._rolling_grad_norms = collections.deque(maxlen=16)

    @classmethod
    def from_args(
            cls,
            training_args,
            distillation_objective_args,
            student_model_args,
            teacher_model_args,
            dataset_args,
            eval_args
    ):
        model_kwargs = {}
        if training_args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif training_args.fp16:
            model_kwargs["torch_dtype"] = torch.float16

        teacher_model, tokenizer = distily.models.get_teacher_model_tokenizer(teacher_model_args, **model_kwargs)
        student_model = distily.models.get_student_model(student_model_args, teacher_model, **model_kwargs)

        # align the max_seq_length to the minimum of the teacher tokenizer / model config and dataset_max_seq_length
        max_seq_len = min(
            tokenizer.model_max_length,
            teacher_model.config.max_position_embeddings,
            dataset_args.dataset_max_seq_length or float("inf")
        )
        tokenizer.model_max_length = max_seq_len
        student_model.config.max_position_embeddings = max_seq_len
        dataset_args.dataset_max_seq_length = max_seq_len

        evaluators = {
            metric["name"]: distily.metrics.get_ppl_metric(tokenizer=tokenizer, **metric)
            for metric in (eval_args.ppl_evaluators + eval_args.ppl_extra_evaluators)
        }

        train_dataset, test_dataset = distily.data.get_dataset(dataset_args, tokenizer)

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
            evaluators=evaluators,
            all_args=dict(
                distillation_objective_args=distillation_objective_args,
                student_model_args=student_model_args,
                teacher_model_args=teacher_model_args,
                dataset_args=dataset_args,
                eval_args=eval_args,
            )
        )

    @classmethod
    def from_kwargs(cls, **kwargs):
        parsed_args_tuple = distily.args.parser.parse_dict(
            kwargs,
            allow_extra_keys=False
        )
        return cls.from_args(*parsed_args_tuple)

    def train(self, *args, **kwargs):
        train_output = super().train(*args, **kwargs)
        if self.args.eval_on_end:
            self.evaluate()
        bench_metrics_out = self._maybe_benchmark()
        if bench_metrics_out is None:
            return train_output
        return transformers.trainer_utils.TrainOutput(
            train_output.global_step,
            train_output.training_loss,
            {**train_output.metrics, **bench_metrics_out}
        )

    def create_optimizer(self):
        """Update optimizer with distillation_objective parameters"""
        optimizer = super().create_optimizer()

        # dry run to initialize the lazy DistillationObjective modules
        with torch.no_grad():
            row = self._remove_unused_columns(self.train_dataset.select(range(1)))[:]
            inputs = {
                k: torch.tensor(v).to(self.model.device)
                for k, v in row.items()
            }
            inputs = {
                k: v.to(dtype=self.model.dtype) if torch.is_floating_point(v) else v
                for k, v in inputs.items()
            }
            self.distillation_objective.forward(self.model, self.teacher_model, inputs)

        # add the named parameters - a bit hacky, involves creating a temporary optimizer
        new_optimizer_params = list(self.distillation_objective.parameters())
        if new_optimizer_params:
            temp_optim = optimizer.__class__(self.distillation_objective.parameters())
            optimizer.param_groups += temp_optim.param_groups
            del temp_optim

        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        # TODO: remove this hack because liger doesn't support labels revert labels deletion once resolved:
        # https://github.com/linkedin/Liger-Kernel/issues/242#issuecomment-2341891320
        del inputs["labels"]

        loss_dict = self.distillation_objective.forward(model, self.teacher_model, inputs)
        loss = loss_dict.pop("loss")

        stats = {k: float(v) for k, v in loss_dict.items()}

        self._extra_stats.append(stats)

        if return_outputs:
            # TODO: real output, this is nothing of use
            return loss, torch.tensor([1.0])
        else:
            return loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        """
        Copy of https://github.com/huggingface/transformers/blob/52a021375/src/transformers/trainer.py#L3394

        With additional behavior:
        - Enable gradient variance logging
        - add self._extra_stats to logs and clear if log step
        """

        ##############
        # NEW CODE
        ##############
        if self.all_args["eval_args"].grad_var_stats:
            self._rolling_grad_norms.append(grad_norm.item())
        ##############
        # END NEW CODE
        ##############

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if transformers.trainer.is_torch_xla_available():
                transformers.trainer.xm.mark_step()

            logs = {}

            ##############
            # NEW CODE
            ##############

            transposed_stats = collections.defaultdict(list)
            [transposed_stats[key].append(d.get(key)) for d in self._extra_stats for key in d]
            for k in transposed_stats:
                if k[0] != "_":
                    logs[k] = sum(transposed_stats[k]) / len(transposed_stats[k])

            if len(self._rolling_grad_norms) == 16 and self.all_args["eval_args"].grad_var_stats:
                logs["grad_norm_var"] = statistics.variance(self._rolling_grad_norms)

            self._extra_stats = []

            ##############
            # END NEW CODE
            ##############

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def evaluate(self, *args, metric_key_prefix="eval", **kwargs):
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        metrics = {}
        if metric_key_prefix == "eval":
            with torch.no_grad():
                for evaluator_name, evaluator in self.evaluators.items():
                    metrics[f"eval_{evaluator_name}"] = float(evaluator(
                        self.model,
                        self.args.per_device_eval_batch_size
                    ))

            self.log(metrics)

        metrics.update(
            super().evaluate(*args, **kwargs)
        )

        self.model.train()
        gc.collect()
        torch.cuda.empty_cache()

        return metrics

    def create_model_card(self, *args, **kwargs):
        super().create_model_card(*args, **kwargs)

        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        model_card = ModelCard.load(model_card_filepath)
        model_card = distily.modelcard.update_model_card(model_card, self)
        model_card.save(model_card_filepath)

    def _maybe_benchmark(self):
        if not self.all_args.get("eval_args") or not self.all_args["eval_args"].harness_benchmarks:
            return

        benchmarks = self.all_args["eval_args"].harness_benchmarks
        limit = self.all_args["eval_args"].harness_benchmark_limit
        bootstrap_iters = self.all_args["eval_args"].harness_benchmark_bootstrap_iters

        self.model.eval()
        self.teacher_model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        with shelve.open(self.benchmarks_shelf) as db:
            if "logs/teacher" not in db:
                db["logs/teacher"] = distily.metrics.run_benchmarks(
                    self.teacher_model, self.tokenizer, benchmarks, limit, bootstrap_iters
                )
            student_metrics = distily.metrics.run_benchmarks(
                self.model, self.tokenizer, benchmarks, limit, bootstrap_iters
            )
            db[self.args.logging_dir] = student_metrics

            # write current run logs
            from tensorboardX import SummaryWriter
            for logging_dir, metrics in db.items():
                writer = SummaryWriter(log_dir=logging_dir)
                for metric_name, metric_value in db[logging_dir].items():
                    writer.add_scalar(f"benchmarks/{metric_name}", metric_value, 0)
                writer.close()

        return student_metrics

    @property
    def benchmarks_shelf(self):
        return os.path.join(self.args.output_dir, "benchmarks.shelve")
