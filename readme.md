# Distily

**In one command, distill an existing model into a smaller or different architecture.**

```
pip install -U git+https://github.com/lapp0/distily
```

### Features
Distily allows you to distill a model with
- Smaller architecture: fewer layers, fewer attention heads, narrower MLP, etc
- Quantized weights: e.g. bitnet
- Distinct architecture: e.g. State-Space models such as Mamba

### Usage

**Minimal Example: `distily_gpt2`**

In this simple example, we create and benchmark a distilled version of `gpt2` with 99% benchmark performance, but 1/2 the size.

Command:
```
python3 -m distily.cli \
    --teacher_model_name_or_path gpt2 \
    --output_dir distily_gpt2 \
    --hub_model_id "distily/distily_gpt2" \
    --push_to_hub True \
```

The [Resulting `distily_gpt2` Model](https://huggingface.co/distily/distily_gpt2) has (TODO: explain metrics).


### Further Reading

TODO: commit the linked docs

**Documentation**
- [Distily Documentation](./docs/index.md)

**Models**
- [Official Distily Models](./docs/official_models.md)
- [All HF Models Created With Distily](https://huggingface.co/models?library=Distily)
