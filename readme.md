Training tool

# TODO

## Package
- [x] eval step
- [x] convert loss to one or many of these metrics https://github.com/DD-DuDa/BitDistiller/blob/master/train/mytrainer.py
- [x] experiment with loss based on hidden states
- [ ] gpt2-distily
- [ ] phi-3-mini-instruct-distily
- [ ] convert command line args to be available via dataclasses like in llama_autotuner
- [ ] package

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

# Install

Install Distily:
```
pip install git+https://github.com/lapp0/distily
```

## Bitnet
To train 1.58b bitnet models, you must install [matmulfreellm](https://github.com/ridgerchu/matmulfreellm) via
```
pip install -U git+https://github.com/ridgerchu/matmulfreellm
```

Bitnet models are trained in an unquantized dtype. If a bitnet model is loaded for training, forward passes are quantized to 1.58b, backwards passes are in the default torch dtype (usually bf16 or fp16).

To compress the model, and reduce vram for inference you must use [BitBlas](https://github.com/microsoft/BitBLAS)

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
