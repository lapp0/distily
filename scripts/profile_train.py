from torch.profiler import profile, record_function, ProfilerActivity
from distily.distillation_trainer import DistillationTrainer


def profile_trainer_train(trainer):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True) as prof:
        with record_function("trainer.train()"):
            result = trainer.train()

    # Print CPU time
    print("CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Print CUDA time
    print("\nCUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Print Memory usage
    print("\nMemory usage:")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # Print Stack trace
    print("\nStack trace:")
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))

    # Export Chrome trace
    prof.export_chrome_trace("train_trace.json")

    return result


def get_gpt2_trainer():
    return DistillationTrainer.from_kwargs(
        student_model_name_or_path="gpt2",
        dataset_sample_size=5_000,
        eval_strategy=None,
        eval_on_start=False,
        eval_on_end=False,
        report_to=None,
    )


if __name__ == "__main__":
    trainer = get_gpt2_trainer()
    profile_trainer_train(trainer)
