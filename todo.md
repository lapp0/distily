# IN "SHIP IT" MODE
Complete these necessary steps for v1.0.0 and initial official models

## Finish Training Setup
- [ ] diagnose why my model is 10% smaller than distilgpt2 when using the same architecture
- [x] LICENSE (AGPL for now)
- [x] Default to OpenRail Variant https://www.licenses.ai/rail-license-generator
- [ ] note in readme about privacy by setting `push_to_hub=False`. "In the spirit of collaboration, ..., for reproducability reasons, ..., however for security reasons you may set `push_to_hub=False`.
- [ ] experiment with sythetic data
- [ ] integrate and test checkpointing
- [ ] bitnet to gguf

## Train Models
- [x] `gpt2` -> `distilgpt2`
- [ ] `smollm-130M` -> `distily_smollm-snug` (30 layers -> 15 layers)
- [ ] `smollm-360M` -> `distily_smollm-tight` (same student size as `_snug`)
- [ ] `smollm-1.3B` -> `distily_smollm-packed` (same student size as `_snug`)
- [ ] `phi-2` (2.8B) -> `distily-phi-2`
- [ ] `Phi-3.5-mini-instruct` (3.8B) -> ?
- [ ] `Phi-3-small-8k-instruct` (7.4B) -> ?
- [ ] share models, new models can mode later

# TODO

- [x] rewrite DistillationObjective as an nn.Module
- [x] ensure optimizer considers DistillationLoss parameters as well
- [x] arg for model max seq length
- [x] KLDiv using Liger
- [ ] ensure attention mask = 0 results in model not being trained on token

- [ ] https://arxiv.org/pdf/2306.13649
- [ ] https://arxiv.org/pdf/2402.03898
- [ ] add contact details to readme?
- [ ] verify dataset.map only happens once, doesn't create redundant mapped datasets increasing disk usage
- [ ] clean up objectives.py by using lazy modules and initialize projectors / norms immediately https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html

## Big Benches
- [x] attn big bench, determine default params
- [x] big bench on synthetic dataset
- [ ] bench warmups
- [ ] bench lr's
- [ ] bench lr schedulers
- [ ] repeat attn benchmarks, but use hs

## v0.5.2 Checkpointing
- [ ] integrate and test checkpointing
- [ ] `FileNotFoundError: [Errno 2] No such file or directory: 'distily_experiments_1M/checkpoint-8000/trainer_state.json'`
## v0.5.3: Benchmark Improvements
- [ ] update end-of-run benchmarks so they run in vllm, not hf
- [ ] report `mean` and `geom_mean` of benchmarks, use `geom_mean` as sweep objective

## v0.6.0: Multi-GPU
- [ ] implement and test multi-GPU training
- [ ] try different techniques (FSDP, DeepSpeed) with various params, on different hardware and profile
- [ ] ensure checkpointing still works

## Necessary for v1.0.0
- [ ] documentation, all TODOs in readme.md
- [ ] model card: include metadata for benchmarks to include evaluation results
- [x] specify datasets by argument
- [x] specify metrics by argument
- [ ] add tooling to convert to 1.58b safetensors file
  - https://github.com/deepsilicon/Sila has it as a WIP I think? https://www.youtube.com/watch?v=yjz84P3UD9g

# Post Release

## Post-v1.0.0 Models
Distily Models:
- [ ] `subtily_small`: Goal is to create the SOTA 1GB model through bitnet, MoE, and distillation from multiple models
  - [Best compression likely involves MoE](https://arxiv.org/html/2404.05567v1)
  - [Best compression likely involves b1.58](https://arxiv.org/abs/2402.17764)

## Post-v1.0.0 Top Priority

### Misc
- [ ] loading the same dataset multiple times increases disk usage: same seed shuffle with same dataset should result in no additional disk usage
- [ ] Complete simple docs
  - ask https://towardsdatascience.com/tinybert-1a928ba3082b for permission to use images

### Optimizations
- [ ] ability to distill bitnet models using 8-bit backward pass, or if there are precision issues, autocast?


## Post-v1.0.0 High Priority

### Misc
- [ ] simple visualization for docs
- [ ] add stochastic noise / batch regularization and experiment since smaller batch size performrs so much better

### Architecture Parameter Sweep
- [ ] bayesian hyperparam search using optuna https://github.com/optuna/optuna/tree/d00b9509b461c24dd0b2dbb2ad8561973d4ad929/tutorial/20_recipes
- [ ] Distill same model to different shapes and sizes, see which architectures can hold the greatest capacity

### Prepare For publishing
- [ ] finish or organize all TODOs in `notes.md`

Organize details of all experiments and perform additional experiments to prepare for engagement with others
- [ ] evaluate results with logit distillation vs hidden state distillation
- [ ] review notes.md, review code, create document overviewing
  - all methods used with sources
  - all novel methods benchmarked and their run script and dashboard
  - all new experiments to run

### Create Issues
Didn't want to tackle these right now, but should create an issue
- [ ] log version of package, including commit in model card
- [ ] include full reproduction steps in model card, including `pip install distily@version` and run command based on params

### Clean Up
- [ ] combine projector and norm, they are deeply intertwined
- [ ] implement `def detailed_forward(outputs)` which calculates arbitrary features (not just those allowed in `model()`)
  - features: `q`, `k`, `v`, `hidden_states`, `attentions`, `attn_output`
  - format: instead of tuples of tuples of tensors, return named tuples of named tensors

## Post-v1.0.0 Medium Priority

### Improved Auditability
- [ ] garbage collect each train round, and each eval round. log train memory and eval memory each step
- [ ] log train and eval time each step

### Reproduce
- [ ] [minitron](https://www.alphaxiv.org/abs/2408.11796v1)

### Optimizations
- [ ] profiling script, use https://github.com/pytorch/kineto/tree/main?tab=readme-ov-file#holistic-trace-analysis
- [ ] [fuse operations in DistillationLoss](https://pytorch.org/tutorials/recipes/fuse.html)
- [ ] use vllm for teacher output, serialize
- [ ] fix whitening functions in ortho projection https://arxiv.org/pdf/2403.06213
  - precompute whitened value across runs since teacher is static
  - good opportunity to introduce data "recording" for teacher

### Synthetic Data
- [ ] currently synthetic data involves full sequence generation. We need to create a synthetic QA / instruct dataset
  - e.g. https://aws.amazon.com/blogs/machine-learning/use-llama-3-1-405b-to-generate-synthetic-data-for-fine-tuning-tasks/
- [ ] [Focus on generating sequences that are likely under the teachers distribution](https://www.semanticscholar.org/paper/GKD%3A-Generalized-Knowledge-Distillation-for-Models-Agarwal-Vieillard/51cda783aa6a97e0b3b5915a2bb5a35f31f3c083)
### Reproduce
- [ ] https://www.alphaxiv.org/abs/2408.11796v1

### Loss Weighting
- [ ] Experiment: weight loss based on teacher model forward pass perplexity
- [ ] Experiment: gradient weighted loss (review paper, see if there's ways to handle case where activations gradients push model in opposite direction as logit gradients / are orthogonal) (see research.md)

### Additional Projectors
- [ ] ensemble projector
- [ ] combine MLP and orthogonal
- [ ] combined attention / hidden projector, use same MLP at bottom, but different projectors on top to take care of diverging dimensionality

### Additional Layer Mappers
- [ ] experiment with "full cross" layer mapper with MLP in pursuit of letting the MLP determine how to apply to each layer
  - [ ] variation: different linear "head-per-layer" to MLP projector for each layer

### Using Adapters
- [ ] train randomly initialized model with ReLoRA
- [ ] Train adapters or existing model with LoRA / DoRA

### Pytorch Lighting
- [ ] Look into rewriting in pytorch lightning - lightning has a lot of the missing features and structure I was seeking in torch + transformers

## Low Priority
- [ ] fix sinkhorn RuntimeError: "cdist_cuda" not implemented for 'BFloat16
- [ ] test mutual_information_loss
