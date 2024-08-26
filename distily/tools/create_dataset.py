import math
import typing
import torch
from dataclasses import dataclass, asdict

"""
TODO: Remove

Might need
pip uninstall -y typing_extensions
pip install typing_extensions==4.11.0
"""


@dataclass
class ExponentialDecayArguments:
    start_t: float = 100.0
    end_t: float = 0.5
    N: int = 1024
    scale_factor: int = 5


@dataclass
class DatasetGenerationArguments:
    model_uri: str
    n_samples: int
    max_length: int
    dataset_uri: str
    private: bool = False
    temperature: typing.Optional[float] = 1.0
    decayed_temperature: typing.Optional[ExponentialDecayArguments] = None
    description_method: str = "Generated sequences randomly with `temperature=1.0`"


class TemperatureDecayLogitsProcessor:
    """
    Logits processor which decays temperature based on sequence length
    - start_t: temperature for 0th token
    - end_t: temperature for Nth token
    - N: max sequence length
    - scale_factor: Increase this to make the distribution drop faster initially
    """
    def __init__(self, decay_args: ExponentialDecayArguments):
        k = (1 / decay_args.N) * torch.log(torch.tensor(decay_args.end_t / decay_args.start_t)) * decay_args.scale_factor
        self.exponential_decay_fn = (
            lambda x: decay_args.start_t * torch.exp(k * x)
        )

    def __call__(self, input_ids, logits):
        temperature = self.exponential_decay_fn(input_ids.shape[-1])
        return logits / temperature


def gen_seq_vllm(args: DatasetGenerationArguments) -> typing.List[str]:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    bs = 64

    llm = LLM(
        args.model_uri,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        n=bs,
        max_tokens=args.max_length,
    )
    if args.decayed_temperature:
        sampling_params.logits_processors = [TemperatureDecayLogitsProcessor(args.decayed_temperature)]
    elif args.temperature:
        sampling_params.temperature = args.temperature
    else:
        raise ValueError("Need temperature or decayed_decayed_temperature")

    responses = llm.generate(
        [llm.get_tokenizer().bos_token] * math.ceil(args.n_samples / bs),
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    return [out.text for r in responses for out in r.outputs]


def create_dataset_card(model_uri, n_samples, max_length, temperature_config):
    from huggingface_hub import DatasetCard

    content = "\n\n".join([
        "# Distillation dataset created with [Distily](https://github.com/lapp0/distily).",
        f"- **Method**: Generated sequences randomly with temperature config `{temperature_config}`",
        f"- **Model URI**: `{model_uri}`",
        f"- **Number of Samples**: {n_samples}",
        f"- **Maximum Sequence Length**: {max_length} tokens",
    ])

    card = DatasetCard(content)
    # card.data["license"] = TODO
    card.data["library_name"] = "Distily"
    card.data["tags"] = ["Distily"]
    card.data["source_datasets"] = ["Original", "Synthetic"]

    return card


def create_empty_dataset_repo_with_description(
    dataset_uri: str,
    model_uri: str,
    n_samples: int,
    max_length: int,
    private: bool,
    temperature: typing.Optional[float],
    decayed_temperature,
) -> None:
    from huggingface_hub import HfApi
    api = HfApi()

    # Create a new repository on the Hugging Face Hub with a description
    api.create_repo(
        repo_id=dataset_uri,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    dataset_card = create_dataset_card(model_uri, n_samples, max_length, temperature or decayed_temperature)
    dataset_card.push_to_hub(dataset_uri)

    print(f"Empty dataset repository {dataset_uri} with dataset card created successfully.")


def create_seq_dataset(args: DatasetGenerationArguments):
    from datasets import Dataset, DatasetDict

    create_empty_dataset_repo_with_description(
        dataset_uri=args.dataset_uri,
        model_uri=args.model_uri,
        n_samples=args.n_samples,
        max_length=args.max_length,
        private=args.private,
        temperature=args.temperature,
        decayed_temperature=args.decayed_temperature,
    )

    sequences = gen_seq_vllm(args)

    dataset = Dataset.from_dict({"text": sequences})
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(args.dataset_uri)


if __name__ == "__main__":
    args = DatasetGenerationArguments(
        dataset_uri="distily/synth_tdecay_gpt2_seq_1K",
        model_uri="gpt2",
        n_samples=1000,
        max_length=1024,
        private=False,
        temperature=None,
        decayed_temperature=ExponentialDecayArguments(),
    )
    create_seq_dataset(args)
