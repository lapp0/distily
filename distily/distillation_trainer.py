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


def _pack_bit_tensor(bool_tensor):
    """Packs a boolean tensor into an int64 tensor using bitwise operations and summation."""
    assert len(bool_tensor.shape) == 1
    padding = (64 - bool_tensor.shape[0] % 64) % 64
    if padding > 0:
        bool_tensor = torch.cat([bool_tensor, torch.zeros(padding, dtype=bool)])

    bit_groups = bool_tensor.view(-1, 64)
    shifts = torch.arange(64, device=bool_tensor.device, dtype=torch.int64)

    packed_tensor = torch.sum(bit_groups * (1 << shifts), dim=-1, dtype=torch.int64)
    return packed_tensor


def _bit_tensor_sum(packed_tensor):
    """Counts the number of 1-bits in a packed int64 tensor using the Hamming weight algorithm."""
    count = packed_tensor
    count = (count - ((count >> 1) & 0x5555555555555555))
    count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333)
    count = (count + (count >> 4)) & 0x0F0F0F0F0F0F0F0F
    count = (count * 0x0101010101010101) >> 56
    return torch.sum(count)


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

        self._prev_grad = None
        self._prev_grad_8bit = None
        self._prev_grad_4bit = None
        self._prev_grad_sign = None
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

        teacher_model, tokenizer = distily.models.get_teacher_model_tokenizer(teacher_model_args)
        student_model = distily.models.get_student_model(student_model_args, teacher_model)

        evaluators = {
            metric["name"]: distily.metrics.get_ppl_metric(tokenizer=tokenizer, **metric)
            for metric in (eval_args.ppl_evaluators + eval_args.ppl_extra_evaluators)
        }

        max_seq_len = min(
            student_model.config.max_position_embeddings,
            teacher_model.config.max_position_embeddings
        )
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
            row = self._remove_unused_columns(self.train_dataset.select(range(1)))[0]
            inputs = {k: torch.tensor(v).to(self.model.device) for k, v in row.items()}
            self.distillation_objective.forward(self.model, self.teacher_model, inputs)

        # add the named parameters - a bit hacky, involves creating a temporary optimizer
        temp_optim = optimizer.__class__(self.distillation_objective.parameters())
        optimizer.param_groups += temp_optim.param_groups
        del temp_optim

        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False, return_stats=False):
        # TODO: remove this hack because liger doesn't support labels revert labels deletion once resolved:
        # https://github.com/linkedin/Liger-Kernel/issues/242#issuecomment-2341891320
        del inputs["labels"]

        loss_dict = self.distillation_objective.forward(model, self.teacher_model, inputs)
        loss = loss_dict.pop("loss")

        stats = {k: float(v) for k, v in loss_dict.items()}

        if return_outputs:
            # TODO: real output, this is nothing of use
            return loss, torch.tensor([1.0])
        elif stats:
            return loss, stats
        else:
            return loss

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Copy of https://github.com/huggingface/transformers/blob/52a021375/src/transformers/trainer.py#L3394

        With additional behavior:
        - Enable comparative gradient logging
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if transformers.trainer.is_sagemaker_mp_enabled():
            loss_mb = transformers.trainer.smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, stats = self.compute_loss(model, inputs, return_stats=True)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if transformers.trainer.is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif transformers.trainer.is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif transformers.trainer.is_torch_musa_available():
                torch.musa.empty_cache()
            elif transformers.trainer.is_torch_npu_available():
                torch.npu.empty_cache()
            elif transformers.trainer.is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [transformers.trainer.OptimizerNames.LOMO, transformers.trainer.OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with transformers.trainer.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        ##############
        # NEW CODE
        ##############

        # add stats
        # add gradient details to
        if self.all_args["eval_args"].binary_grad_similarity_stats:
            grad_sign = torch.cat([_pack_bit_tensor(p.grad.flatten() > 0) for p in model.parameters()])
            if self._prev_grad_sign is not None:
                sign_xor = grad_sign ^ self._prev_grad_sign
                stats["grad_bin_prev_similarity"] = 1 - (_bit_tensor_sum(sign_xor).item() / (sign_xor.numel() * 64))
            self._prev_grad_sign = grad_sign
        if self.all_args["eval_args"].full_grad_similarity_stats:
            flat_grad = [p.grad.to(torch.float16).view(-1) for p in model.parameters()]
            if self._prev_grad is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    dot_product = sum(torch.dot(curr, prev) for curr, prev in zip(flat_grad, self._prev_grad))
                    norm1 = sum(curr.norm() ** 2 for curr in flat_grad)
                    norm2 = sum(prev.norm() ** 2 for prev in self._prev_grad)
                stats["grad_prev_cos_sim"] = (dot_product / (norm1**0.5 * norm2**0.5)).item()
            self._prev_grad = flat_grad

            """
            import bitsandbytes as bnb
            flat_grad_8bit = list(map(bnb.functional.quantize_blockwise, flat_grad))
            if self._prev_grad_8bit is not None:
                dot_product = sum(torch.dot(curr, prev).item() for curr, prev in zip(flat_grad_8bit, self._prev_grad_8bit))
                norm1 = sum(curr.norm().item() ** 2 for curr in flat_grad_8bit)
                norm2 = sum(prev.norm().item() ** 2 for prev in self._prev_grad_8bit)
                stats["grad_prev_cos_sim"] = dot_product / (norm1**0.5 * norm2**0.5)
            self._prev_grad_8bit = flat_grad_8bit

            flat_grad_8bit = list(map(bnb.functional.quantize_4bit, flat_grad))
            if self._prev_grad_4bit is not None:
                dot_product = sum(torch.dot(curr, prev).item() for curr, prev in zip(flat_grad_4bit, self._prev_grad_4bit))
                norm1 = sum(curr.norm().item() ** 2 for curr in flat_grad_4bit)
                norm2 = sum(prev.norm().item() ** 2 for prev in self._prev_grad_4bit)
                stats["grad_prev_cos_sim"] = dot_product / (norm1**0.5 * norm2**0.5)
            self._prev_grad_4bit = flat_grad_4bit
            """
            #gc.collect()
            #torch.cuda.empty_cache()

        self._extra_stats.append(stats)

        ##############
        # END NEW CODE
        ##############

        return loss.detach() / self.args.gradient_accumulation_steps

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        """
        Copy of https://github.com/huggingface/transformers/blob/52a021375/src/transformers/trainer.py#L3394

        With additional behavior:
        - Enable gradient variance logging
        - clear self._extra_stats once queued for logging
        """
        self._rolling_grad_norms.append(grad_norm.item())
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
            if "teacher" not in db:
                db["teacher"] = distily.metrics.run_benchmarks(
                    self.teacher_model, self.tokenizer, benchmarks, limit, bootstrap_iters
                )
            student_metrics = distily.metrics.run_benchmarks(
                self.model, self.tokenizer, benchmarks, limit, bootstrap_iters
            )
            db[self.args.run_name] = student_metrics

        return student_metrics

    @property
    def benchmarks_shelf(self):
        return os.path.join(self.args.output_dir, "benchmarks.shelve")
