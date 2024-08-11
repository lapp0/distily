Training tool

TODO: explanation and examples


# Install

Install Distily:
```
pip install -U git+https://github.com/lapp0/distily
```

Install default dependencies
```
pip install -U bitsandbytes tensorboardX flash-attn
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
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
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
- Cross-Entropy
- Sinkhorn Loss

## Features / Knowledge Sources
- Logits: Evidenced as a more efficient knowledge source
- ...
- ...
- Attentions: MiniLM trained last attention layers KL divergence

**TODO: What methods exist for feature knowledge**
- https://www.researchgate.net/profile/Jianping-Gou/publication/342094012/figure/fig10/AS:950418217648129@1603608769145/The-generic-instance-relation-based-knowledge-distillation.png
- Different relation-based features include FSP Matrix, Instance Relation, Similarity matrix, Representation graph etc.


# Papers

## A Comparative Analysis of Task-Agnostic Distillation Methods for Compressing Transformer Language Models
https://arxiv.org/pdf/2310.08797

Explores simple logits distillation, along with hidden state and attention layer mapping strategies.

Model Features
- logits: uses cross entropy
- MHA: "distil the attention relation matrices (Q-Q, K-K and V-V) obtained by first concatenating the query (Q), key (K), and value (V) mappings from all attention heads and re-splitting them into the same number of attention relation heads"
- hidden states: using mean square error

Layer Mapping Strategies
- 1-to-1: Maps student layers to uniformly distributed teacher layers.
  - Last: simple, but effective baseline
  - Last K uniform: `teacher_layer_i` -> `student_layer_i`
  - All: Requires identical num layers
- 1-to-N Mapping: Maps each student layer to multiple teacher layers.
  - Uniform Consecutive: Maps each student layer to k consecutive teacher layers.
    - Formula: ϕ(i)=[k(i−1),ki]ϕ(i)=[k(i−1),ki], where k=⌈LT/LS⌉k=⌈LT​/LS​⌉.
	- Ensures all teacher layers contribute to the student's learning.
  - Uniform + Last: Each student layer maps to two teacher layers: one selected uniformly and one from the last layers.
    - Combines both Uniform and Last strategies from 1-to-1 mapping.
	- Each student layer maps to two teacher layers: one selected uniformly and one from the last layers.
	- Leverages the benefits of capturing both early syntactic features and late semantic features.

Loss Functions:
- Cross Entropy
- MHA MSE: sum over (Q, K, V) and attention heads of MSE(student_relation_matrix, teacher_relation_matrix)
- Direct MHA MSE: sum over (Q, K, V) and attention heads of MSE(student_mapping * W, teacher_mapping)
- MSE: Sum of MSE

Linear transformation:
- The loss functions learn a linear transformation. This is only necessary when the dimensions vary.

Experiments:
- Logits
  - layer mapping strategy: N/A
  - loss: cross entropy
- Hidden States:
  - Layer Mapping Strategies: 1-to-1, Uniform Consecutive, and Uniform + Last are used.
  - loss: MSE
- Attentions:
  - Layer Mapping Strategy: Single mapping strategy, focusing on aligning attention weights directly.
  - Loss: MHA MSE, Direct MHA MSE

Results:
- "For both MiniLMv2 and DirectMiniLM, we found distilling the upper-middle teacher layer, i.e. (LT −1)th or (LT −2)th strategy,"
- "Importantly, we found that both MHA transfer methods generally outperform HS transfer, which points to the benefit of transferring the Q/K/V knowledge over the hidden state knowledge."
- Direct MHA MSE didn't consistently improve performance based on their provided data. When it did, improvement wasn't substantial. Therefore we should only implement MHA MSE.

## Distiller: A Systematic Study of Model Distillation Methods in Natural Language Processing
https://www.semanticscholar.org/reader/08460ecff91b8a54358b9c1709d7dc6a77417f62

Core parameters:
- data augmentation policy
- a layer mapping policy for intermediate distillation
  - skip: the student learns from every [M / N] layer of the teacher
  - last: the student learns from the last k layers of the teacher
  - EMD: a many-to-many learned layer mapping strategy (Li et al., 2020) based on Earth Mover’s Distance
- intermediate distillation objective
  - Cross-Entropy (CE)
  - Mean Squared Error (MSE)
  - L2 distance, Cosine Similarity (Cos)
  - Patient Knowledge Distillation (PKD)
  - Mutual Information (MI-alpha)
- logits objective
  - Cross-Entropy (CE)
  - Mean Squared Error (MSE)

"n Iα, f (·, ·) and q(·) are critic functions forapproximating unknown densities and m(·, ·) is aMonte-Carlo estimate of the partition function thatappears in MI calculations. Typically, the space zand the sample x, y are from the same minibatchwhile training, that is K +1 equals to the minibatchsize. Iα can flexibly trade off bias and variance,since increasing α ∈ [0, 1] will reduce the vari-ance of the estimator while increasing its bias. Wepropose to use Iα as an objective for intermediatedistillation and call it MI-α. Our implementationleverages a Transformer encoder (Vaswani et al.,2017) to learn f (·, ·) and q(·). To our knowledge, this is the first attempt to utilize complex NN archi- tectures for critic functions in MI estimation; typi- cally only shallow multilayer perceptrons (MLPs) are used (Tschannen et al., 2020). Our experiments (Table 4 in Appendix) reveal that Transformer pro- duces a better critic function than MLP".

"objectives like MI (and tighter bounds thereof) merely attempt to ensure the information in the teacher rep- resentation is also captured in the student represen- tation. The latter aim is conceptually better suited for KD, particularly in settings where the student’s architecture differs from the teacher"

Innovations in KD for NLP generally involve improvements in one of the following aspects:
1) the loss function for gauging the discrepancy between student and teacher predictions
2) the method for transferring intermediate network representations between teacher and student
3) the use of data augmentation during student training
4) multiple stages of distillation


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

Eval Performance of distance functions (consistent across datasets):
- best: SinKD
- 2nd: Jensen-Shannon (JS)
- 3rd: Reverse-KL
- 4th: KL

implemented as `distily.objectives.sinkhorn_loss`

"Existing KDmethods investigate various divergence measures including the Kullback-Leibler (KL), reverse Kullback-Leibler(RKL), and Jensen-Shannon (JS) divergences. However, due to limitations inherent in their assumptions anddefinitions, these measures fail to deliver effective supervision when few distribution overlap exists betweenthe teacher and the student."

"e propose the Sinkhorn Knowledge Distillation (SinKD) that exploits the Sinkhorn distance to ensure a nuanced and precise assessment of the disparity between teacher and student distributions."

## Understanding and Improving Knowledge Distillation for Quantization Aware Training of Large Transformer Encoders
https://arxiv.org/pdf/2211.11014

## One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation
https://www.semanticscholar.org/reader/b39d324da2b6b728334c52927885c0e10494c935

## Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression
https://arxiv.org/pdf/2310.00369

## Understanding the Effects of Projectors in Knowledge Distillation
https://arxiv.org/pdf/2310.17183

TODO

"Conventionally, during the knowledge distillation process (e.g. feature distillation), an additional projector is often required to perform feature transformation due to the dimension mismatch between the teacher and the student networks. Interestingly, we discovered that even if the student and the teacher have the same feature dimensions, adding a projector still helps to improve the distillation performance"

## Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs
https://arxiv.org/abs/2402.12030

## Gradient Knowledge Distillation for Pre-trained Language Models
https://arxiv.org/abs/2211.01071

## Improving Knowledge Distillation for BERT Models: Loss Functions, Mapping Methods, and Weight Tuning
https://arxiv.org/pdf/2308.13958

## PET: Parameter-efficient Knowledge Distillation on Transformer
https://www.semanticscholar.org/paper/PET%3A-Parameter-efficient-Knowledge-Distillation-on-HyojinJeon-Park/e01204e05440881e5edcd6a872fa40e3f8474a89

## Revisiting Intermediate Layer Distillation for Compressing Language Models: An Overfitting Perspective
https://www.semanticscholar.org/paper/Revisiting-Intermediate-Layer-Distillation-for-An-Ko-Park/21a5cd656e6d1426d46c443fb85a41bc2dc53bef

## KS-DETR: Knowledge Sharing in Attention Learning for Detection Transformer
https://www.semanticscholar.org/paper/KS-DETR%3A-Knowledge-Sharing-in-Attention-Learning-Zhao-Ukita/1ef697894bc8b7321cf4a960e07daf59013d3ea0

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
