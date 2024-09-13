"""
Methods:
1) From text dataset, dynamic
2) From text dataset, preload
3) Synthetic, preload
4) GOLD, dynamic

# Synthetic, preload

Memory per sample:
(note, this includes one "subsample" per token)
- attentions: n_layers * n_heads * n_tokens^2
- hidden_states: (n_layers + 1) * n_tokens * n_embed
- logits: n_tokens * vocab_size

## Example: gpt2 1000 token sample

>>> out = model(torch.tensor([range(1000)]), output_attentions=True, output_hidden_states=True
>>> sys.getsizeof(out.logits.untyped_storage())
201028072
>>> sys.getsizeof(torch.vstack(out.attentions).untyped_storage())
576000072
>>> sys.getsizeof(torch.vstack(out.hidden_states).untyped_storage())
39936072
>>> (201028072+576000072+39936072) / 1024**2
779.117790222168

779 MB per sample, not ideal for many use cases


# GOLD dynamic:
Generate data at runtime based on energy function.

"""

import os
import datasets


def do_tokenize(examples, tokenizer, column_name, **tokenize_kwargs):
    return tokenizer(examples[column_name], **tokenize_kwargs)


def get_dataset(dataset_args, tokenizer):
    dataset = datasets.load_dataset(
        dataset_args.dataset_uri,
        dataset_args.dataset_subset,
        split=dataset_args.dataset_split,
        trust_remote_code=dataset_args.dataset_trust_remote_code,
    )
    if dataset_args.dataset_shuffle:
        dataset = dataset.shuffle(seed=dataset_args.dataset_shuffle_seed)
    dataset = dataset\
        .select(range(dataset_args.dataset_sample_size))\
        .train_test_split(test_size=dataset_args.dataset_test_size)

    ### TODO: REMOVE THIS ONCE STABILITY OF HASH FUNCTION IS VERIFIED
    ### TODO: ALSO VERIFY THAT THE HASH CHANGES IF TOKENIZER CHANGES
    print(f"tokenizer: {tokenizer}")
    print(f"Hash of tokenizer: {datasets.fingerprint.Hasher.hash(tokenizer)}")
    print(f"Hash of tokenize_function: {datasets.fingerprint.Hasher.hash(do_tokenize)}")
    ##################

    tokenized_dataset = dataset.map(
        do_tokenize,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            truncation=True,
            padding="max_length",
            max_length=dataset_args.dataset_max_seq_length,
            column_name=dataset_args.dataset_column_name,
        ),
        batched=True,
        batch_size=100,
        num_proc=os.cpu_count() * 3 // 4,
    )

    ### TODO: REMOVE THIS ONCE STABILITY OF HASH FUNCTION IS VERIFIED
    ### TODO: ALSO VERIFY THAT THE HASH CHANGES IF TOKENIZER CHANGES
    print(f"AFTER: tokenizer: {tokenizer}")
    print(f"AFTER: Hash of tokenizer: {datasets.fingerprint.Hasher.hash(tokenizer)}")
    print(f"AFTER: Hash of tokenize_function: {datasets.fingerprint.Hasher.hash(do_tokenize)}")
    ##################


    return tokenized_dataset["train"], tokenized_dataset["test"]
