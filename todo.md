# TODO

## Model Generation Milestones
- [ ] distill 99% quality model with identical architecture
- [ ] half layer gpt2, but beat distilgpt2 benchmarks
- [ ] distill phi-3-mini to 1.58b, report metrics


## v0.3.0
- [ ] log all training parameters (excluding stuff like push_to_hub)
- [ ] log dataset total token count
- [ ] eval for HotpotQA, TriviaQA, GLUE, SQUAD, CoNLL-2003, CoLA, MNLI
- [ ] fix log output so the loss/logits and loss/activations respects logging_steps
## v0.4.0
Complete basic objectives implementation

**Objectives**
### Implement layer mapping strategies from https://arxiv.org/pdf/2310.08797
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

### Implement loss functions from https://arxiv.org/pdf/2310.08797
Loss Functions:
- Cross Entropy
- MHA MSE: sum over (Q, K, V) and attention heads of MSE(student_relation_matrix, teacher_relation_matrix)
- Direct MHA MSE
- MSE: Sum of MSE

## v0.5.0
Implement synthetic datasets
- randomly sampled
- custom generators focusing on OOD sequences


## Research Task Before v0.5.0
- [ ] Research Data-Free Knowledge Distillation techniques: ensure model is more representative and sample-efficient

## v0.5.1
- [ ] add eval tool for MMLU-PRO / MATH / etc

**Training Quality Improvements**
- [ ] add ability to transfer / freeze embeddings
- [ ] gradient weighted loss (review paper, see if there's ways to handle case where activations gradients push model in opposite direction as logit gradients / are orthogonal)
- [ ] add stochastic noise / batch regularization and experiment since smaller batch size performrs so much better

**Bug Fix**
- [ ] loading the same dataset multiple times increases disk usage
- [ ] fix checkpointing: `FileNotFoundError: [Errno 2] No such file or directory: 'distily_experiments_1M/checkpoint-8000/trainer_state.json'`

**Auditability Improvements**
- [ ] garbage collect each train round, and each eval round. log train memory and eval memory each step
- [ ] log train and eval time each step


## v0.6.0
**Optimizations**
- [ ] use vLLM to prepare base model forward passes
- [ ] use torch 2.5.0 to compile forward pass

- [ ] training qwen-0.5B

## v0.7.0
- [ ] research dataset which would be best for this task

## Necessary for v1.0.0
- [ ] documentation, all TODOs in readme.md
- [ ] model card: include metadata for benchmarks to include evaluation results
- [ ] specify datasets by argument
- [ ] specify metrics by argument
- [ ] add tooling to convert to 1.58b safetensors file
- [ ] fix sinkhorn RuntimeError: "cdist_cuda" not implemented for 'BFloat16
- [ ] test mutual_information_loss


## Create Issues
Didn't want to tackle these
- [ ] log version of package, including commit in model card


## Publish
- [ ] evaluate results with logit distillation vs hidden state distillation
- [ ] search for optimal hidden states to select
- [ ] other hyperparameter experiments

## Optimize
- [ ] ability to distill models using 8-bit backward pass
- [ ] use vllm for teacher output, serialize
