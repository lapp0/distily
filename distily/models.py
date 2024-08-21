import copy
import functools

import transformers
import torch

from distily.bitnet_utils import convert_to_bitnet


def _reinitialize_weights(model, weight_init_fn="xavier"):
    """Reinitialize the weights using the provided weight initialization function."""

    print("reinitializing weights as", weight_init_fn)
    # TODO: full impl
    assert weight_init_fn == "xavier"
    init_fn = torch.nn.init.xavier_uniform_

    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.dim() >= 2:  # Only initialize if the weight tensor has 2 or more dimensions
                init_fn(module.weight)
            else:
                print("skipping, not 2D for", module)


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


def get_teacher_model_tokenizer(teacher_model_args):
    model = transformers.AutoModelForCausalLM.from_pretrained(
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(teacher_model_args.teacher_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_student_model(student_model_args, teacher_model):
    if student_model_args.student_model_name_or_path:
        student_model = transformers.AutoModelForCausalLM.from_pretrained(
            student_model_args.student_model_name_or_path,
            torch_dtype=torch.bfloat16,
        )

    else:
        if student_model_args.student_config_name_or_path:
            config = transformers.AutoConfig.from_pretrained(student_model_args.student_config_name_or_path)
        else:
            config = copy.copy(teacher_model.config)

        if student_model_args.student_model_config:
            config.update(student_model_args.student_model_config)

        # Force student to have vocabulary size as teacher
        config.vocab_size = teacher_model.config.vocab_size

        # TODO: remove .to(...) hack
        student_model = transformers.AutoModelForCausalLM.from_config(
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).to(device="cuda")

    if student_model_args.dropout:  # TODO: argument
        student_model.hidden_dropout_prob = student_model_args.dropout
        student_model.attention_probs_dropout_prob = student_model_args.dropout

    if student_model_args.reinitialize_weights:
        _reinitialize_weights(student_model, student_model_args.reinitialize_weights)

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