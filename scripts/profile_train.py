from torch.profiler import profile, record_function, ProfilerActivity, schedule
from distily.distillation_trainer import DistillationTrainer
from transformers import TrainerCallback


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_context = record_function("TrainingStep")
        self.step_context.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        self.step_context.__exit__(None, None, None)
        self.prof.step()


def profile_trainer_train(trainer):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(skip_first=4, wait=0, warmup=2, active=4, repeat=1),
        with_stack=True,
    ) as prof:
        trainer.add_callback(ProfCallback(prof=prof))
        trainer.train()

        print("\nTop 40 operations by CPU time:")
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=40,
            max_src_column_width=150,
            top_level_events_only=False
        ))

        print("\nTop 40 operations by CUDA time:")
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
            max_src_column_width=150,
            top_level_events_only=False
        ))

        # Export Chrome trace
        prof.export_chrome_trace("train_trace.json")


def get_trainer(**replacements):
    trainer_kwargs = dict(
        teacher_model_name_or_path="HuggingFaceTB/SmolLM-135M",
        dataset_sample_size=64,
        per_device_train_batch_size=4,
        eval_strategy="no",
        eval_on_start=False,
        eval_on_end=False,
        report_to=None,
        output_dir="./profile_distily",
    )
    trainer_kwargs.update(replacements)
    return DistillationTrainer.from_kwargs(**trainer_kwargs)


if __name__ == "__main__":
    trainer = get_trainer(optim="lion_32bit")
    profile_trainer_train(trainer)
