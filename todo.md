# IN "SHIP IT" MODE
Complete these necessary steps for v1.0.0 and initial official models

## Finish Training Setup
- [ ] diagnose why my model is 10% smaller than distilgpt2 when using the same architecture
- [x] LICENSE (AGPL for now)
- [x] Default to OpenRail Variant https://www.licenses.ai/rail-license-generator
- [ ] note in readme about privacy by setting `push_to_hub=False`. "In the spirit of collaboration, ..., for reproducability reasons, ..., however for security reasons you may set `push_to_hub=False`.
- [ ] experiment with sythetic data
- [ ] integrate and test checkpointing

## Train Models
- [x] gpt2 -> distilgpt2
- [ ] qwen-0.5B bitnet, report metrics
- [ ] phi-3-mini_short_betnet, report metrics
- [ ] phi-3-mini_bitnet, report metrics
- [ ] llama-3.1_bitnet, report metrics
- [ ] share models, new models can mode later


# TODO

## Big Benches
- [x] attn big bench, determine default params
- [ ] big bench on synthetic dataset
- [ ] bench warmups
- [ ] bench lr's
- [ ] bench lr schedulers
- [ ] repeat attn benchmarks, but use hs

## v0.5.1 Fix and Benchmark Synthetic Datasets
- [ ] update temperature scaler, temp = 10 for initial token, 0.5 for all remaining tokens
- [ ] ensure dynamic temperature always ends at end_t
- [ ] experiment with training on different decay schedules

## v0.5.2 Checkpointing
- [ ] integrate and test checkpointing
- [ ] `FileNotFoundError: [Errno 2] No such file or directory: 'distily_experiments_1M/checkpoint-8000/trainer_state.json'`
## v0.5.3:
- [ ] update benchmarks so they run in vllm, not hf

## v0.5.4: Improved Auditability
- [ ] garbage collect each train round, and each eval round. log train memory and eval memory each step
- [ ] log train and eval time each step


## v0.5.5: Optimizations
MAYBE skip
- [ ] use torch 2.5.0 to compile forward pass
- [ ] test liger


## Necessary for v1.0.0
- [ ] documentation, all TODOs in readme.md
- [ ] model card: include metadata for benchmarks to include evaluation results
- [x] specify datasets by argument
- [x] specify metrics by argument
- [ ] add tooling to convert to 1.58b safetensors file

## Post Release
Reorganize these
- [ ] simple visualization for docs
- [ ] loading the same dataset multiple times increases disk usage
- [ ] ensemble projector
- [ ] gradient weighted loss (review paper, see if there's ways to handle case where activations gradients push model in opposite direction as logit gradients / are orthogonal) (see research.md)
- [ ] add stochastic noise / batch regularization and experiment since smaller batch size performrs so much better
- [ ] fix whitening functions in ortho projection https://arxiv.org/pdf/2403.06213
  - precompute whitened value across runs since teacher is static
  - good opportunity to introduce data "recording" for teacher
- [ ] fix sinkhorn RuntimeError: "cdist_cuda" not implemented for 'BFloat16
- [ ] test mutual_information_loss
- [ ] bayesian hyperparam search using optun for hidden states

### Architecture Parameter Sweep
- Distill same model to different shapes and sizes, see which architectures can hold the greatest capacity

### Create Issues
Didn't want to tackle these
- [ ] log version of package, including commit in model card


### Publish
- [ ] evaluate results with logit distillation vs hidden state distillation


## Optimize
- [ ] ability to distill bitnet models using 8-bit backward pass
- [ ] use vllm for teacher output, serialize
