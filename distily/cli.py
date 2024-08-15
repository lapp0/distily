import distily

import functools
import copy
from dataclasses import asdict
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

    if teacher_model_args.teacher_model_compile:
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_args.teacher_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def _transfer_module_to_student(student_model, teacher_model, module_name, freeze=False):
    """
    Replace module in student_model with module from teacher model.
    Optionally freeze by disabling requires_grad.
    """
    get_module = lambda model, module_name: functools.reduce(getattr, module_name.split("."), model)

    student_module = get_module(student_model, module_name)
    teacher_module = get_module(teacher_model, module_name)
    student_module.load_state_dict(teacher_module.state_dict())

    # ensure transfer successful
    sm_sd = student_module.state_dict()
    tm_sd = teacher_module.state_dict()
    assert all(torch.equal(sm_sd[k], tm_sd[k]) for k in student_module.state_dict())

    # freeze module
    if freeze:
        for param in get_module(student_model, module_name).parameters():
            param.requires_grad = False


def get_student_model(student_model_args, teacher_model):
    if student_model_args.student_model_name_or_path:
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_args.student_model_name_or_path,
            torch_dtype=torch.bfloat16,
        )

    else:
        if student_model_args.student_config_name_or_path:
            config = AutoConfig.from_pretrained(student_model_args.student_config_name_or_path)
        else:
            config = copy.copy(teacher_model.config)

        if student_model_args.student_model_config:
            config.update(student_model_args.student_model_config)

        # Force student to have vocabulary size as teacher
        config.vocab_size = teacher_model.config.vocab_size

        # TODO: remove .to(...) hack
        student_model = AutoModelForCausalLM.from_config(
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")

    for module_name, freeze in (student_model_args.copy_teacher_modules or []):
        _transfer_module_to_student(student_model, teacher_model, module_name=module_name, freeze=freeze)

    if student_model_args.student_model_as_bitnet:
        with torch.no_grad():
            # TODO: use a different method which is better supported, an official third party library
            convert_to_bitnet(student_model, copy_weights=False)
            student_model.model_tags = ["bitnet", "1.58b"]

    if student_model_args.student_model_compile:
        student_model.forward = torch.compile(student_model.forward, mode="reduce-overhead", fullgraph=True)

    return student_model



def get_dataset(dataset_args, tokenizer, max_seq_len: int):
    dataset = datasets.load_dataset(
        dataset_args.dataset_uri,
        dataset_args.dataset_subset,
        split=dataset_args.dataset_split
    )
    dataset = dataset\
        .select(range(dataset_args.dataset_sample_size))\
        .train_test_split(test_size=dataset_args.dataset_test_size)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x[dataset_args.dataset_column_name],
            truncation=True,
            padding="max_length",
            max_length=max_seq_len
        ),
        batched=True,
        batch_size=100,
        num_proc=os.cpu_count() * 3 // 4,
    )
    return tokenized_dataset["train"], tokenized_dataset["test"]


def get_distillation_objective(distillation_objective_args):
    kwargs = asdict(distillation_objective_args)
    return distily.objectives.DistillationObjective(**kwargs)


def do_train(training_args, distillation_objective_args, student_model_args, teacher_model_args, dataset_args):

    # TODO: don't hardcode max length
    max_seq_len = 1024

    teacher_model, tokenizer = get_teacher_model_tokenizer(teacher_model_args)
    student_model = get_student_model(student_model_args, teacher_model)
    train_dataset, test_dataset = get_dataset(dataset_args, tokenizer, max_seq_len)

    # TODO: don't hardcode this
    training_args.extra_evaluators = distily.metrics.get_all_metric_evaluators(tokenizer)

    # TODO: cleanup, use a get_distillation_objective() fn
    distillation_objective = get_distillation_objective(distillation_objective_args)

    trainer = distily.distillation_trainer.DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        distillation_objective=distillation_objective,
    )

    ##############
    # TODO: REMOVE
    print(trainer.eval_teacher_metrics())
    ##############

    trainer.train()

    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    do_train(*distily.args.get_args())
