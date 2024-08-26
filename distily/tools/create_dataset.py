import typing
import torch
from dataclasses import dataclass, asdict


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
    batch_size: int
    dataset_uri: str
    private: bool = False
    temperature: typing.Optional[float] = 1.0
    decayed_temperature: typing.Optional[ExponentialDecayArguments] = None
    description_method: str = "Generated sequences randomly with `temperature=1.0`"


class ExponentialDecayTemperatureLogitsProcessor:
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

    def process_logits(self, input_ids, logits):
        temperature = self.exponential_decay_fn(input_ids.shape[-1])
        return logits / temperature


def gen_seq_vllm(args: DatasetGenerationArguments) -> typing.List[str]:
    from vllm import LLM
    llm = LLM(args.model_uri)

    if args.decayed_temperature:
        sampling_kwargs = {
            "logits_processors": [ExponentialDecayTemperatureLogitsProcessor(**asdict(args.decayed_temperature))]
        }
    elif args.temperature:
        sampling_kwargs = {
            "temperature": args.temperature
        }
    else:
        raise ValueError("Need temperature or decayed_decayed_temperature")

    sequences = llm.generate(
        prompt=[],
        n_samples=args.n_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        **sampling_kwargs
    )
    return sequences


def create_empty_dataset_repo_with_description(
    dataset_uri: str,
    model_uri: str,
    n_samples: int,
    max_length: int,
    private: bool = False
) -> None:
    from huggingface_hub import HfApi
    api = HfApi()

    # TODO: don't hardcode
    method = "Generated sequences randomly with `temperature=1.0`"

    # Create a description for the dataset
    description = "\n\n".join([
        "# Distillation dataset created with [Distily](https://github.com/lapp0/distily).",
        f"- **Method**: {method}",
        f"- **Model URI**: {model_uri}\n"
        f"- **Number of Samples**: {n_samples}\n"
        f"- **Maximum Sequence Length**: {max_length}\n"
    ])

    # Create a new repository on the Hugging Face Hub with a description
    api.create_repo(
        repo_id=dataset_uri,
        repo_type="dataset",
        private=private,
        repo_description=description
    )

    print(f"Empty dataset repository {dataset_uri} with description created successfully.")


def create_seq_dataset(args: DatasetGenerationArguments):
    from datasets import Dataset, DatasetDict

    create_empty_dataset_repo_with_description(
        dataset_uri=args.dataset_uri,
        model_uri=args.model_uri,
        n_samples=args.n_samples,
        max_length=args.max_length,
        private=args.private
    )

    sequences = gen_seq_vllm(args)

    dataset = Dataset.from_dict({"text": sequences})
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(args.dataset_uri)


if __name__ == "__main__":
    args = DatasetGenerationArguments(
        dataset_uri="distily/synthetic_gpt2_sequences_1K",
        model_uri="gpt2",
        n_samples=1000,
        max_length=1024,
        batch_size=16,
        private=False,
        temperature=1.0,
        decayed_temperature=None
    )
    create_seq_dataset(args)
