"""
import click
from datasets import load_dataset
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from tinyllama_bitnet_utils import convert_to_bitnet

from distillation_trainer import DistillationTrainer
import metrics


def initialize_bitnet_from_base_model(base_model_uri):
    base_config = AutoConfig.from_pretrained(base_model_uri)
    base_config.attn_implementation = "flash_attention_2"
    model = AutoModelForCausalLM.from_config(base_config)\
                                .to(dtype=torch.bfloat16)\
                                .to("cuda:0")
    with torch.no_grad():
        convert_to_bitnet(model, copy_weights=False)
    model.model_tags = ["bitnet", "1.58b"]
    return model


@click.command()
@click.option("--dataset_name", default="wikimedia/wikipedia", help="Name of the dataset to load.")
@click.option("--teacher_uri", default="microsoft/Phi-3-mini-4k-instruct", help="URI of the teacher model.")
def run(dataset_name, teacher_uri):

    # TODO: Remove hardcode
    teacher_uri = "gpt2"

    # TODO: don't hardcode dataset
    dataset = load_dataset(dataset_name, "20231101.en", split="train[:5000000]")
    dataset = dataset.train_test_split(test_size=0.01)

    tokenizer = AutoTokenizer.from_pretrained(teacher_uri)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    student_model = initialize_bitnet_from_base_model(base_model_uri=teacher_uri)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_uri,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    teacher_model.eval()

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
        ),
        batched=True
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    output_model_name = "gpt2_distily"

    # TODO: from args based on tuner
    trainer_args = TrainingArguments(
        output_dir=f"./{output_model_name}",
        hub_model_id=f"lapp0/{output_model_name}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="steps",
        eval_steps=2000,
        logging_steps=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        lr_scheduler_type="cosine",
        learning_rate=1e-4,
        max_grad_norm=64.0,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_steps=2000,
        push_to_hub=True,
        report_to="tensorboard",
        save_total_limit=5,
        eval_on_start=True,
    )

    all_extra_metric_evaluators = metrics.get_all_metric_evaluators(tokenizer)

    trainer = DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        extra_evaluators=all_extra_metric_evaluators,
        activation_loss_pairs=True,
    )

    # log base model metrics
    base_model_results = {}
    with torch.no_grad():
        for evaluator_name, evaluator in all_extra_metric_evaluators.items():
            base_model_results[f"eval_{evaluator_name}"] = evaluator(trainer.teacher_model)
        base_model_results["epoch"] = 0
        trainer.log(base_model_results)

    trainer.train()

    if trainer_args.push_to_hub:
        trainer.push_to_hub()
"""

from . import args as distily_args
from . import distillation_trainer
from . import metrics as distily_metrics

import datasets
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling


def get_teacher_model_tokenizer(teacher_model_args):
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_args.teacher_model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=teacher_model_args.teacher_load_in_8bit,
        load_in_4bit=teacher_model_args.teacher_load_in_4bit,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_args.teacher_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_student_model(student_model_args, teacher_model_args):
    if student_model_args.student_model_as_bitnet:
        from mmfreelm.models import HGRNBitForCausalLM, HGRNBitConfig

    if student_model_args.student_model_name_or_path:
        model_cls = HGRNBitForCausalLM if student_model_args.student_model_as_bitnet else AutoModelForCausalLM
        return model_cls.from_pretrained(student_model_args.student_model_name_or_path)
    else:
        config_uri = student_model_args.student_model_name_or_path or teacher_model_args.teacher_model_name_or_path
        config = AutoConfig.from_pretrained(config_uri)
        if student_model_args.student_model_config:
            config.update(student_model_args.student_model_config)
        if student_model_args.student_model_as_bitnet:
            config = HGRNBitConfig(config)
        config.attn_implementation = "flash_attention_2"
        config._attn_implementation = "flash_attention_2"
        return HGRNBitForCausalLM.from_config(config).to(dtype=torch.bfloat16)


def run():
    training_args, student_model_args, teacher_model_args = distily_args.get_args()

    teacher_model, tokenizer = get_teacher_model_tokenizer(teacher_model_args)
    student_model = get_student_model(student_model_args, teacher_model_args)

    teacher_model = teacher_model.cuda()
    student_model = student_model.cuda()

    # TODO: don't hardcode dataset
    #train_dataset = get_train_dataset(dataset_args)
    #test_dataset = get_test_dataset(dataset_args)
    #extra_metrics = get_ppl_eval_datasets(dataset_args)
    dataset = datasets.load_dataset("wikipedia/wikipedia", "20231101.en", split="train[:5000000]")
    dataset = dataset.train_test_split(test_size=0.01)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=tokenizer.model_max_length),
        batched=True
    )
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["test"]
    extra_metric_evaluators = distily_metrics.get_all_metric_evaluators(tokenizer)

    trainer = distillation_trainer.DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        extra_evaluators=extra_metric_evaluators,  # TODO
        activation_loss_pairs=True,  # TODO
    )

    if True: #args.log_teacher_metrics
        trainer.log_teacher_metrics()

    trainer.train()

    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    run()
