import distily

import os
import copy
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling


# TODO: REMOVE THIS
from .tinyllama_bitnet_utils import convert_to_bitnet


def get_teacher_model_tokenizer(teacher_model_args):
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_args.teacher_model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=teacher_model_args.teacher_load_in_8bit,
        load_in_4bit=teacher_model_args.teacher_load_in_4bit,
        device_map="cuda"
    )
    # freeze (maybe redundant)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_args.teacher_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_student_model(student_model_args, teacher_config):
    if student_model_args.student_model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            student_model_args.student_model_name_or_path,
            torch_dtype=torch.bfloat16,
        )

    else:
        if student_model_args.student_config_name_or_path:
            config = AutoConfig.from_pretrained(student_model_args.student_config_name_or_path)
        else:
            config = copy.copy(teacher_config)

        if student_model_args.student_model_config:
            config.update(student_model_args.student_model_config)

        # Force student to have vocabulary size as teacher
        config.vocab_size = teacher_config.vocab_size

        # TODO: remove .to(...) hack
        model = AutoModelForCausalLM.from_config(
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")

    if student_model_args.student_model_as_bitnet:
        with torch.no_grad():
            # TODO: use a different method which is better supported, an official third party library
            convert_to_bitnet(model, copy_weights=False)
            model.model_tags = ["bitnet", "1.58b"]

    return model


def run():
    training_args, student_model_args, teacher_model_args = distily.args.get_args()

    teacher_model, tokenizer = get_teacher_model_tokenizer(teacher_model_args)

    student_model = get_student_model(student_model_args, teacher_model.config)

    # TODO: don't hardcode max length
    max_seq_len = 1024

    # TODO: don't hardcode dataset
    #train_dataset = get_train_dataset(dataset_args)
    #test_dataset = get_test_dataset(dataset_args)
    #extra_metrics = get_ppl_eval_datasets(dataset_args)
    dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    dataset = dataset.select(range(1000000))
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_seq_len),
        batched=True,
        batch_size=100,
        num_proc=os.cpu_count() * 3 // 4,
    )
    train_dataset = tokenized_dataset
    #test_dataset = tokenized_dataset["test"]
    # TODO: don't hardcode this
    training_args.extra_evaluators = distily.metrics.get_all_metric_evaluators(tokenizer)

    trainer = distily.distillation_trainer.DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tprain_dataset=train_dataset,
        #eval_dataset=test_dataset,
    )

    trainer.train()

    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    run()
