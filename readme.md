Training tool

# TODO

## Priorities
- [ ] add base model eval to model card
- [ ] add distilled model full metrics table to model card
- [ ] add tooling to convert to 1.58b safetensors file
- [ ] distill phi-3-mini to 1.58b
- [ ] add eval tool for MMLU / MATH /  etc
- [ ] add ability to transfer / freeze embeddings


## Package
- [x] eval step
- [x] convert loss to one or many of these metrics https://github.com/DD-DuDa/BitDistiller/blob/master/train/mytrainer.py
- [x] experiment with loss based on hidden states
- [x] gpt2-distily
- [ ] phi-3-mini-instruct-distily
- [x] convert command line args to be available via dataclasses like in llama_autotuner
- [x] package

## Publish
- [ ] evaluate results with logit distillation vs hidden state distillation
- [ ] search for optimal hidden states to select
- [ ] look into freezing embeddings, experiment
- [ ] other hyperparameter experiments

## Clean
- [ ] model card - description of method
- [ ] model card - remove incorrect "finetuned from"
- [X] model card - verify tags
- [ ] model card - ensure original model and new model metrics included
- [ ] use vllm for teacher output, serialize

## Optimize
- [ ] ability to distill models using 8-bit backward pass

# Install

Install Distily:
```
pip install -U git+https://github.com/lapp0/distily
```

## Bitnet
Bitnet models are trained in an unquantized dtype. If a bitnet model is loaded for training, forward passes are quantized to 1.58b, backwards passes are in the default torch dtype (usually bf16 or fp16).

To compress the model, and reduce vram for inference you must use [BitBlas](https://github.com/microsoft/BitBLAS)

# Usage

## CLI

(Trained on 1 x RTX 4090)
```
python3 -m distily.cli \
    --teacher_model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --output_dir qwen2-0.5-experiments_control \
    --hub_model_id "lapp0/qwen2-0.5-experiments_control" \
    --push_to_hub True \
    --report_to tensorboard
```

<details>


(Trained on 1 x A100 40GB)

(TODO)
```
python3 -m distily.cli \
    --teacher_model_name_or_path microsoft/Phi-3-mini-4k-instruct \
    --student_model_as_bitnet True \
    --output_dir phi-3-mini-4k-instruct_distily \
    --hub_model_id "lapp0/phi-3-mini-4k-instruct_distily" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 2000 \
    --logging_steps 4 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate 5e-5 \
    --max_grad_norm 64.0 \
    --gradient_checkpointing True \
    --optim paged_adamw_32bit \
    --save_steps 2000 \
    --push_to_hub True \
    --report_to tensorboard \
    --eval_on_start True \
    --teacher_load_in_8bit True
```

</details>


<details>


(Trained on 1 x A100 40GB)

(TODO)
```
python3 -m distily.cli \
    --teacher_model_name_or_path gpt2 \
    --student_model_as_bitnet False \
    --output_dir gpt2_bf16_distily \
    --hub_model_id "lapp0/gpt2_bf16_distily" \
    --per_device_train_batch_size 16 \
    --eval_strategy steps \
    --eval_steps 2000 \
    --logging_steps 4 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate 5e-5 \
    --max_grad_norm 64.0 \
    --gradient_checkpointing True \
    --optim paged_adamw_32bit \
    --save_steps 2000 \
    --push_to_hub True \
    --report_to tensorboard \
    --eval_on_start True
```

</details>


<details>


(Trained on 1 x RTX 4090)
link to examples

</details>


## Python

TODO

# Hardware Requirements

TODO: expand

Requires memory sufficient for loading both teacher and student model, forward pass on both, backprop on student.

bitnet models require at least bf16 or fp16 memory allocation during training.


# Modules
- `distill_bitnet.py` (runner)
- `distillation_trainer.py` (logit distiller by default)
- `teacher_feature_generator.py` (runs vllm and generates logits)

# Experiments

## Overview

Models:
- gpt2
- phi-3-mini

Training sets:
- "wikimedia/wikipedia" "20231101.en"
- random text

Parameters
- No Activations, Select Activations, All Activations
- Learning Rate
- Loss fn
- Number of Samples
- Number of Epochs
- Base model quantization level

Eval metrics:
- Loss
- MATH
- MMLU-PRO
- Wikipedia PPL on Articles after training cutoff
- Wikipedia PPL on Articles in other languages

## GPT2

Activations: None, lr: 4e-4, loss_fn: reverse_kl, samples: 45000, epochs: 3, base quant: BNB4, loss:


# Resources

## Example bitnet trainer
https://github.com/Ferix-Inc/ferix-ai-blog/blob/main/BitNet/bitnet.ipynb

## Bitnet Trainer for Simple Distillation
Distils based on sampled token, not based on anything more advanced like logits or activations:
https://github.com/DD-DuDa/BitDistiller/

## Transformers Output with Intermediate Activations
https://github.com/huggingface/transformers/blob/c1aa0edb48217f416f4bbe6e3a9db1500284513b/src/transformers/models/dpt/modeling_dpt.py#L58

## Response-based vs Feature-based distillation
https://neptune.ai/blog/knowledge-distillation
