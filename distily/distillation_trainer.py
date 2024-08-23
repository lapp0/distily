from dataclasses import asdict
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

    def from_kwargs(cls, **kwargs):
        parsed_args_tuple = distily.args.parser.parse_dict(
            kwargs,
            allow_extra_keys=True
        )
        return cls.from_args(*parsed_args_tuple)

    def train(self, *args, **kwargs):
        train_output = super().train(*args, **kwargs)
        bench_metrics_out = self._maybe_benchmark()
        if bench_metrics_out is None:
            return train_output
        return transformers.trainer_utils.TrainOutput(
            train_output.global_step,
            train_output.training_loss,
            {**train_output.metrics, **bench_metrics_out}
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_dict = self.distillation_objective(self.teacher_model, model, inputs)
        loss = loss_dict.pop("loss")

        # if train step, add to log history
        if not return_outputs:
            self.state.log_history.append({
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
        model_card.data["library_name"] = "Distily"
        if self.all_args.get("dataset_args"):
            model_card.data["datasets"] = [self.all_args["dataset_args"].dataset_uri]

        model_card.text = distily.modelcard.create_model_card_text(self)

        model_card.save(model_card_filepath)

    def _maybe_benchmark(self):
        if not self.all_args.get("eval_args") or not self.all_args["eval_args"].harness_benchmarks:
            return

        benchmarks = self.all_args["eval_args"].harness_benchmarks

        self.model.eval()
        self.teacher_model.eval()
        gc.collect()
        torch.cuda.empty_cache()

        with shelve.open(self.benchmarks_shelf) as db:
            if "teacher" not in db:
                db["teacher"] = distily.metrics.run_benchmarks(
                    self.teacher_model, self.tokenizer, benchmarks
                )
            student_metrics = distily.metrics.run_benchmarks(
                self.model, self.tokenizer, benchmarks
            )
            db[self.args.run_name] = student_metrics

        return student_metrics

    @property
    def benchmarks_shelf(self):
        return os.path.join(self.args.output_dir, "benchmarks.shelve")
