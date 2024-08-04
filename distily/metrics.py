from transformers import TrainerCallback
from datasets import load_dataset
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


# TODO: Respect eval batch size for `batch_size` argument
class PerplexityEvalCallback(TrainerCallback):
    def __init__(self, dataset, tokenizer, batch_size=1, max_length=1024, dataset_column="text"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        # preprocess / tokenize
        predictions = [example[dataset_column] for example in self.dataset]
        self.encodings = self.tokenizer(
            predictions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def do_eval(self, model):
        input_ids = self.encodings["input_ids"].to(model.device)
        attention_mask = self.encodings["attention_mask"].to(model.device)

        loss_fct = CrossEntropyLoss(reduction="none")
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for start_index in tqdm(range(0, len(input_ids), self.batch_size)):
                end_index = start_index + self.batch_size
                batch_input_ids = input_ids[start_index:end_index]
                batch_attention_mask = attention_mask[start_index:end_index]

                outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits

                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch_input_ids[..., 1:].contiguous()

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                mask = batch_attention_mask[..., 1:].contiguous().view(-1)
                loss = (loss * mask).sum()

                total_loss += loss.item()
                total_count += mask.sum().item()

        avg_loss = total_loss / total_count
        # Convert avg_loss to a tensor before computing perplexity
        avg_loss_tensor = torch.tensor(avg_loss)
        perplexity = torch.exp(avg_loss_tensor).item()
        return perplexity


def get_all_metric_evaluators(tokenizer):
    return {
        "enwikippl": PerplexityEvalCallback(
            dataset=load_dataset("wikimedia/wikipedia", "20231101.en", split="train[-1000:]"),
            tokenizer=tokenizer,
        ).do_eval,
        "frwikippl": PerplexityEvalCallback(
            dataset=load_dataset("wikimedia/wikipedia", "20231101.fr", split="train[-1000:]"),
            tokenizer=tokenizer,
        ).do_eval,
        "zhwikippl": PerplexityEvalCallback(
            dataset=load_dataset("wikimedia/wikipedia", "20231101.zh", split="train[-1000:]"),
            tokenizer=tokenizer,
        ).do_eval,
    }
