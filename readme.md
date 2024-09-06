# Distily


#### In one command, distill an existing LLM into a smaller or different architecture.


## Install

```
pip install -U "git+https://github.com/lapp0/distily.git#egg=distily[full]"
```

## Features
Distily allows you to distill a model with
- Quantized weights: e.g. TriLM, bitnet
- Distinct architecture: State-Space models such as Mamba, Mixture-of-Experts (MoE)
- Modified architecture: Decrease (or increase) the
  - number of layers
  - width and depth of attention heads and dense layer.
  - the number of attention and KV heads.

## Usage

**Minimal Example: `distily_gpt2`**

Command to create a distilled `gpt2` with only 6 layers:
```
python3 -m distily.run \
    --teacher_model_name_or_path gpt2 \
    --output_dir distily_gpt2 \
    --hub_model_id "distily/distily_gpt2" \
    --push_to_hub True \
    --student_model_config {"n_layers": 6}
```

The [Resulting `distily_gpt2` Model](https://huggingface.co/distily/distily_gpt2) has (TODO: explain metrics).

For more examples, review the [Examples](./docs/examples.md) documentation.

#### Note on Hub Credentials
To push to hub, you must prepare your hub token
```
HF_WRITE=<your hub token> python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('${HF_WRITE}')"
```

## Further Reading

TODO: commit the linked docs once complete

**Using Distily**
- How Distillation Works: [The Distily Recipe](./docs/recipe.md)
- [Quickstart / Examples](./docs/using.md)
- [Parameter Selection](./docs/params.md)

**Available Models**
- [Official Distily Models](./docs/official_models.md)
- [All HF Models Created With Distily](https://huggingface.co/models?library=Distily)


**Contributing**
- [Contributing Guidelines](./docs/contributing.md)

## Roadmap

#### Improved performance / sampling efficiency:
- [X] Standard knowledge distillation using logits.
- [x] Distill using intermediate features including hidden states and attentions.
- [ ] Improve sampling efficiency through synthetic data generation.
- [ ] Implement cross-entropy classification loss (traditional LLM loss function)
- [ ] Apply projector to logits (https://arxiv.org/pdf/2310.17183)
- [ ] Apply "teacher recording", run teacher inference once, use features dataset any number of times.

#### Distill to a different model shape / size:
- [x] Distill to model with fewer `num_hidden_layers` by implementing layer mappers.
- [x] Distill to a model with modified module dimensions and behaviors (e.g., `intermediate_size`, `hidden_act`) by employing projectors.
- [x] Distill to a model with modified `num_attention_heads` and `num_key_value_heads`.

#### Distill to a different architecture:
- [x] Distill to Bitnet (b1.58)
- [ ] Distill to State-Space / Mamba
- [ ] Distill to MoE
- [ ] Distill to Parameter Sharing (ALBERT-style) Model

#### Additional Techniques:
- [ ] Distill from multiple models at once
