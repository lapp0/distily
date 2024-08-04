import datasets
from transformers import DataCollatorForLanguageModeling

from distily.cli import get_teacher_model_tokenizer, get_student_model
from distily.args import StudentModelArguments, TeacherModelArguments, DistillationTrainingArguments
from distily import metrics
from distily import distillation_trainer


def run():
    teacher_model_args = TeacherModelArguments(teacher_model_name_or_path="microsoft/Phi-3-mini-4k-instruct")
    teacher_model, tokenizer = get_teacher_model_tokenizer(teacher_model_args)

    student_model_args = StudentModelArguments(
        student_config_name_or_path="gpt2",
        student_model_as_bitnet=True,
        student_model_config={"hidden_size": teacher_model.config.hidden_size},
    )

    student_model = get_student_model(student_model_args, teacher_model_args, teacher_model.vocab_size)

    # TODO: don't hardcode dataset
    #train_dataset = get_train_dataset(dataset_args)
    #test_dataset = get_test_dataset(dataset_args)
    #extra_metrics = get_ppl_eval_datasets(dataset_args)
    dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:10000]")
    dataset = dataset.train_test_split(test_size=0.01)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=1024  # set lower for rtx 4090
        ),
        batched=True,
        batch_size=10000,
        num_proc=8,
    )
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]

    training_args = DistillationTrainingArguments(
        output_dir="phi-3-mini-4k-instruct_distily_striped_activations",
        hub_model_id="lapp0/phi-3-mini-4k-instruct_distily_striped_activations",
        per_device_train_batch_size=1,  # set low for rtx 4090
        eval_strategy="steps",
        eval_steps=2000,
        logging_steps=4,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        max_grad_norm=64.0,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_steps=2000,
        push_to_hub=True,
        report_to="tensorboard",
        eval_on_start=True,
    )
    training_args.extra_evaluators = metrics.get_all_metric_evaluators(tokenizer)

    trainer = distillation_trainer.DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        activation_loss_pairs=[(32, 12), (30, 11), (16, 8), (8, 4), (4, 3), (2, 2), (1, 1), (0, 0)],
    )

    trainer.train()

    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    run()
