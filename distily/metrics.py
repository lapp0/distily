from transformers import TrainerCallback
from datasets import load_dataset
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


class PerplexityEvalCallback(TrainerCallback):
    def __init__(self, dataset, tokenizer, max_length=1024, dataset_column="text", add_start_token=True):
        # preprocess / tokenize
        predictions = [example[dataset_column] for example in dataset]
        self.encodings = tokenizer(
            predictions,
            padding=True,
            truncation=True,
            max_length=max_length - 1 if add_start_token else max_length,
            return_tensors="pt"
        )
        if add_start_token:
            bos_token_ids = torch.full((self.encodings["input_ids"].size(0), 1), tokenizer.bos_token_id)
            self.encodings["input_ids"] = torch.cat([bos_token_ids, self.encodings["input_ids"]], dim=1)
            self.encodings["attention_mask"] = torch.cat([
                torch.ones_like(bos_token_ids),
                self.encodings["attention_mask"]
            ], dim=1)

    def do_eval(self, model, batch_size):
        input_ids = self.encodings["input_ids"].to(model.device)
        attention_mask = self.encodings["attention_mask"].to(model.device)

        total_loss = 0.0
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            for start_index in range(0, len(input_ids), batch_size):
                end_index = min(start_index + batch_size, len(input_ids))
                batch_input_ids = input_ids[start_index:end_index]
                batch_attention_mask = attention_mask[start_index:end_index]

                # Generate model outputs
                outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits

                # Shift logits and labels to calculate loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch_input_ids[..., 1:].contiguous()
                shift_attention_mask_batch = batch_attention_mask[..., 1:].contiguous()

                # Calculate loss
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = (loss * shift_attention_mask_batch.view(-1)).sum() / shift_attention_mask_batch.sum()

                total_loss += loss.item() * shift_attention_mask_batch.sum().item()
                total_tokens += shift_attention_mask_batch.sum().item()

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
            dataset=load_dataset("roneneldan/TinyStories", None, split="validation").select(range(2000)),
            tokenizer=tokenizer,
        ).do_eval,
    }
