Training tool

# TODO

## Priorities
- [x] add base model eval to model card
- [x] add distilled model full metrics table to model card
- [x] get working qwen2-0.5b run
- [ ] Change DistillationStrategy to DistillationObjective, which handles the entire loss calculation
- [ ] eval for HotpotQA, TriviaQA, GLUE, SQUAD, CoNLL-2003
- [ ] add eval tool for MMLU / MATH / etc
- [ ] add ability to transfer / freeze embeddings
- [ ] fix re-entrant issue

## Necessary for v1.0.0
- [ ] model card: include metadata for benchmarks to include evaluation results
- [ ] specify datasets by argument
- [ ] specify metrics by argument
- [ ] add tooling to convert to 1.58b safetensors file
- [ ] distill phi-3-mini to 1.58b, report metrics

## Clean
- [x] model card - description of method
- [x] model card - remove incorrect "finetuned from"
- [X] model card - verify tags
- [x] model card - ensure original model and new model metrics included

## Package
- [x] eval step
- [x] convert loss to one or many of these metrics https://github.com/DD-DuDa/BitDistiller/blob/master/train/mytrainer.py
- [x] experiment with loss based on hidden states
- [x] gpt2-distily
- [x] convert command line args to be available via dataclasses like in llama_autotuner
- [x] package

## Publish
- [ ] evaluate results with logit distillation vs hidden state distillation
- [ ] search for optimal hidden states to select
- [ ] other hyperparameter experiments

## Optimize
- [ ] ability to distill models using 8-bit backward pass
- [ ] use vllm for teacher output, serialize

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
    --output_dir distily_experiments_control \
    --hub_model_id "lapp0/distily_experiments_control" \
    --push_to_hub True \
    --report_to tensorboard \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8
    --teacher_load_in_8bit True
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


# Methods
There are a number of approaches for consideration

## Loss Function
- KL Divergence: The typical loss function for intermediate features
- MSE: Two different papers evidence the strength of MSE as an alternative
- Cosine Similarity: One paper suggests cosine similarity of the fetaures is effective.
#
## Features / Knowledge Sources
- Logits: Evidenced as a more efficient knowledge source
- ...
- ...
- Attentions: MiniLM trained last attention layers KL divergence

**TODO: What methods exist for feature knowledge**
- https://www.researchgate.net/profile/Jianping-Gou/publication/342094012/figure/fig10/AS:950418217648129@1603608769145/The-generic-instance-relation-based-knowledge-distillation.png
- Different relation-based features include FSP Matrix, Instance Relation, Similarity matrix, Representation graph etc.

# Papers

## A Survey on Transformer Compression
https://arxiv.org/pdf/2402.05964
Reviews hint-based (activations, attentions) and logit-based distillation methods

"MobileBERT implements two objective functions to distill knowledge from a BERT teacher incorporated with inverted bottleneck, including attention distributions and hidden states, to a slimmed-down version of BERT as the student model"

In "Well-read students learn better: The impact of student initialization on knowledge distillation." they find that tiny models with pretraining learn better.

## A Closer Look at Knowledge Distillation with Features, Logits, and Gradients
https://arxiv.org/pdf/2203.10163

Paper covers CNN, not transformers, but provides a number of insights on objective function design.

Most effective knowledge sources ranked
- Logits
- features weighted by gradients
- plain features


**Other Details**

"he Fisher information F (zt), which leverages the gradi- ents regarding the teacher’s intermediate representation, provides a weighting mechanism for the importance of features."

"Note z can be the logits or features. Besides, the knowledge in the teacher’s gradients are transferred to the student via W (zt). Therefore LKD−G provides a unified framework for comparing the effectiveness of each knowledge source by in- stantiating it in different ways"

"logits are generally a more efficient knowledge source and suggests that having sufficient feature dimensions is crucial for the model design, providing a practical guideline for effective KD-based trans- fer learning."

"Heo et al. (2019b) shows the features are more effective; Tian, Krishnan, and Isola (2020) observes that logits are generally better, but matching the pairwise correlation with features outperforms logits."

"Kim et al. (2021) reports that logits achieves a better result when using L2 loss instead of KL-divergence"


## MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers
https://arxiv.org/pdf/2002.10957

Objective function: `L = KL(AT_teacher, AT_student) + KL(VR_teacher, VR_student)`

where
- AT: attention distribution (SDP of Q, K)
- VR: value relation (SDP of V, V)

This method has the benefit of not constrain the size or number of layers.

## Efficient Transformer Knowledge Distillation: A Performance Review
https://arxiv.org/pdf/2311.13657

Uses three loss functions
- unpervised training loss: standard masked language modeling loss
- distillation loss: cross entropy over soft targets
- hidden state loss: (based on DistilBERT) cosine embedding loss between student/techear hidden states vectors

(Rest of paper not strictly relevant)

## Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation
https://arxiv.org/pdf/2105.08919

Paper covers WRN (image classification) model, not transformers.

Describes two loss functions and their nature:
- logit matching (KL with high temperature, equivalent to MSE)
  - performs best in their strong teacher experiment
- label matching (KL with low temperature)
  - performs best in their **noisy** teacher experiment.

(Cho and Hariharan, 2019) shows sequential KD (large network → medium network → small network) is not conducive to gen- eralization

## Sinkhorn Distance Minimization for Knowledge Distillation
https://www.semanticscholar.org/reader/7304666dce9e90861ca7de928dd6826fb338fbd8

## Distiller: A Systematic Study of Model Distillation Methods in Natural Language Processing
https://www.semanticscholar.org/reader/08460ecff91b8a54358b9c1709d7dc6a77417f62

## Understanding and Improving Knowledge Distillation for Quantization Aware Training of Large Transformer Encoders
https://arxiv.org/pdf/2211.11014

## One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation
https://www.semanticscholar.org/reader/b39d324da2b6b728334c52927885c0e10494c935

## Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression
https://arxiv.org/pdf/2310.00369

# Resources



## Use MSE for activation loss
(Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation, 2021)

## Example bitnet trainer
https://github.com/Ferix-Inc/ferix-ai-blog/blob/main/BitNet/bitnet.ipynb

## Bitnet Trainer for Simple Distillation
Distils based on sampled token, not based on anything more advanced like logits or activations:
https://github.com/DD-DuDa/BitDistiller/

## Transformers Output with Intermediate Activations
https://github.com/huggingface/transformers/blob/c1aa0edb48217f416f4bbe6e3a9db1500284513b/src/transformers/models/dpt/modeling_dpt.py#L58

## Response-based vs Feature-based distillation
https://neptune.ai/blog/knowledge-distillation
