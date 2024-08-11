# TODO

## Model Generation Milestones
- [ ] distill 99% quality model with identical architecture
- [ ] distill phi-3-mini to 1.58b, report metrics

## v0.2.1
- [ ]

## v0.3.0
**Auditability Improvements**
- [ ] log version of package, including commit
- [ ] garbage collect each train round, and each eval round. log train memory and eval memory each step
- [ ] log train and eval time each step
- [ ] eval for HotpotQA, TriviaQA, GLUE, SQUAD, CoNLL-2003, CoLA, MNLI

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
- Direct MHA MSE: (do not implement)
- MSE: Sum of MSE


## v0.5.0
- [ ] add eval tool for MMLU / MATH / etc
- [ ] log all training parameters (excluding stuff like push_to_hub)

**Training Quality Improvements**
- [ ] add ability to transfer / freeze embeddings
- [ ] gradient weighted loss (review paper, see if there's ways to handle case where activations gradients push model in opposite direction as logit gradients / are orthogonal)

**Bug Fix**
- [ ] loading the same dataset multiple times increases disk usage
- [ ] fix checkpointing: `FileNotFoundError: [Errno 2] No such file or directory: 'distily_experiments_1M/checkpoint-8000/trainer_state.json'`

## Necessary for v1.0.0
- [ ] model card: include metadata for benchmarks to include evaluation results
- [ ] specify datasets by argument
- [ ] specify metrics by argument
- [ ] add tooling to convert to 1.58b safetensors file
- [ ] fix sinkhorn RuntimeError: "cdist_cuda" not implemented for 'BFloat16
- [ ] test mutual_information_loss


## v1.1.0
- [x] benchmark with https://github.com/huggingface/transformers/issues/14608


## v0.2.0 (Next Experiment Set)
- [x] Change DistillationStrategy to DistillationObjective, which handles the entire loss calculation
- [x] fix re-entrant issue
- [x] model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
- [x] allow dataset initialization by training arguments


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
