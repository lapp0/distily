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


def get_teacher_model_tokenizer(teacher_model_args, **model_kwargs):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        teacher_model_args.teacher_model_name_or_path,
        load_in_8bit=teacher_model_args.teacher_load_in_8bit,
        load_in_4bit=teacher_model_args.teacher_load_in_4bit,
        device_map="cuda",
        **model_kwargs
    )

    # freeze (maybe redundant)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(teacher_model_args.teacher_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if teacher_model_args.model_max_length is not None:
        tokenizer.model_max_length = teacher_model_args.model_max_length

    return model, tokenizer


def get_student_model(student_model_args, teacher_model, **model_kwargs):
    if student_model_args.student_use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        # automodel_cls = AutoLigerKernelForCausalLM
        # TODO: remove hack below, use above comment once https://github.com/linkedin/Liger-Kernel/issues/242 is fixed
        class PatchedAutoLiger(AutoLigerKernelForCausalLM):
            @staticmethod
            def from_config(config, *args, **kwargs):
                AutoLigerKernelForCausalLM.from_pretrained(config._name_or_path)
                return AutoLigerKernelForCausalLM.from_config(config, *args, **kwargs)
        automodel_cls = PatchedAutoLiger

    else:
        automodel_cls = transformers.AutoModelForCausalLM

    if student_model_args.student_model_name_or_path:
        student_model = automodel_cls.from_pretrained(
            student_model_args.student_model_name_or_path,
            device_map="cuda",
            **model_kwargs
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

        config.update(model_kwargs)

        # TODO: remove .to(...) hack, use autocast
        config.use_cache = False
        student_model = automodel_cls.from_config(
            config=config,
        ).to("cuda")

    if student_model_args.reinitialize_weights:
        _reinitialize_weights(student_model, student_model_args.reinitialize_weights)

    for module_name, freeze in (student_model_args.copy_teacher_modules or []):
        _transfer_module_to_student(student_model, teacher_model, module_name=module_name, freeze=freeze)

    if student_model_args.student_model_as_bitnet:
        with torch.no_grad():
            # TODO: use a different method which is better supported, an official third party library
            convert_to_bitnet(student_model, copy_weights=False)
            student_model.model_tags = ["bitnet", "1.58b"]

    return student_model
