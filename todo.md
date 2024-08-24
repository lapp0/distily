# Model Generation Milestones
- [ ] distill 99% quality model with identical architecture
- [ ] half layer gpt2, but beat distilgpt2 benchmarks
- [ ] distill phi-3-mini to 1.58b, report metrics


# TODO



## v0.4.0
- [ ] Implement synthetic datasets: randomly sampled

## v0.4.1
performance:
- [ ] replace `student/teacher_model_compile` with `training_args.compile`
- [ ] integrate liger kernel https://github.com/linkedin/Liger-Kernel
- [ ] smoke test `torch.compile`
- [ ] smoke test `torch.optim._multi_tensor` https://github.com/huggingface/transformers/issues/9965
- [ ] smoke test with `dynamo.optimize("inductor")`

## v0.4.2
- [ ] `debug` mode: more overhead and slower, but can help debug loss functions. Calculate the following:
      - grad_norm of each individual loss component (determine magnitude)
	  - correlation between each loss components gradient (determine relatedness)

## v0.4.3
- [ ] batch norm layer independent of projector

## v0.5.0
- [ ] synthetic datasets: custom generators focusing on OOD sequences with vllm
- [ ] update benchmarks so they run in vllm, not hf

## v0.5.1
- [ ] benchmark_harness for HotpotQA, TriviaQA, GLUE, SQUAD, CoNLL-2003, CoLA, MNLI
  - [ ] additional: MMLU-PRO / MATH / etc

## v0.5.2
- [ ] fix log output so the loss/logits and loss/activations respects logging_steps


## v0.5.3
**Training Quality Improvements**
- [x] add ability to transfer / freeze embeddings
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
