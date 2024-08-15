from transformers import TrainerCallback
from datasets import load_dataset
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


# TODO: Respect eval batch size for `batch_size` argument
class PerplexityEvalCallback(TrainerCallback):
    def __init__(self, dataset, tokenizer, max_length=1024, dataset_column="text"):
        # preprocess / tokenize
        predictions = [example[dataset_column] for example in dataset]
        self.encodings = tokenizer(
            predictions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def do_eval(self, model, batch_size):
        input_ids = self.encodings["input_ids"].to(model.device)
        attention_mask = self.encodings["attention_mask"].to(model.device)

        total_loss = 0.0
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            for start_index in tqdm(range(0, len(input_ids), batch_size)):
                end_index = start_index + batch_size
                batch_input_ids = input_ids[start_index:end_index]
                batch_attention_mask = attention_mask[start_index:end_index]

                outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_input_ids)
                loss = outputs.loss * batch_attention_mask.sum()  # Sum of non-padded tokens in the batch

                total_loss += loss.item()
                total_tokens += batch_attention_mask.sum().item()  # Sum of non-padded tokens in the batch

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity


def get_all_metric_evaluators(tokenizer):
    return {
        "enwikippl": PerplexityEvalCallback(
            dataset=load_dataset("wikimedia/wikipedia", "20231101.en", split="train").select(range(2000000, 2001000)),
            tokenizer=tokenizer,
        ).do_eval,
        "frwikippl": PerplexityEvalCallback(
            dataset=load_dataset("wikimedia/wikipedia", "20231101.fr", split="train").select(range(1000)),
            tokenizer=tokenizer,
        ).do_eval,
        "zhwikippl": PerplexityEvalCallback(
            dataset=load_dataset("wikimedia/wikipedia", "20231101.zh", split="train").select(range(1000)),
            tokenizer=tokenizer,
        ).do_eval,
        "tinystoriesppl": PerplexityEvalCallback(
            dataset=load_dataset("roneneldan/TinyStories", None, split="validation"),
            tokenizer=tokenizer,
        ).do_eval,
    }
