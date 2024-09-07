import json
from datasets import load_dataset


class DocumentFilter:
    def __init__(self, **kwargs):
        # Default filter parameters
        self.filters = {
            "all_caps_words_max": 0.1,
            "unique_words_range": (0.1, 0.9),
            "mean_word_length_range": (3, 10),
            "lines_end_with_ellipsis_max": 0.2,
            "bullet_point_lines_max_ratio": 0.9,
            "no_alph_words_max": 0.4,
            "stop_word_fraction_min": 0.06,
            "symbol_to_word_ratio_max": 0.5,
            "word_count_range": (50, 10000),
            #"ccnet_bucket_max": 0.9,
            #"ccnet_perplexity_max": 487.5,
            "curly_bracket_max": 0.05,
            "lorem_ipsum_max": 3e-06,
            #"books_importance_range": (-float("inf"), float("inf")),
            #"openwebtext_importance_min": -float("inf"),
            "unigram_entropy_range": (3, 6),
            "top_2_gram_frac_max": 0.2,
            "top_3_gram_frac_max": 0.18,
            "top_4_gram_frac_max": 0.16,
            "dupe_5_gram_frac_max": 0.15,
            "dupe_6_gram_frac_max": 0.14,
            "dupe_7_gram_frac_max": 0.13,
            "dupe_8_gram_frac_max": 0.12,
            "dupe_9_gram_frac_max": 0.11,
            "dupe_10_gram_frac_max": 0.1,
            "language": None,
            "allow_dupe": False,
        }

        # Update filters with provided kwargs
        self.filters.update(kwargs)

        # Initialize counters to track how many documents fail each filter
        self.fail_counters = {key: 0 for key in self.filters}

    def in_range(self, val, min_max):
        """Helper method to check if a value is within a range."""
        if val is None:
            return False
        return min_max[0] <= val <= min_max[1]

    def __call__(self, sample):
        """Applies the filters to a sample and increments the counters for failed filters."""
        signals = json.loads(sample["quality_signals"])
        meta = json.loads(sample["meta"])
        # Rule evaluation using a dictionary of checks
        rule_checks = {
            "all_caps_words_max": signals["rps_doc_frac_all_caps_words"][0][2] < self.filters["all_caps_words_max"],
            "unique_words_range": self.in_range(signals["rps_doc_frac_unique_words"][0][2], self.filters["unique_words_range"]),
            "mean_word_length_range": self.in_range(signals["rps_doc_mean_word_length"][0][2], self.filters["mean_word_length_range"]),
            "lines_end_with_ellipsis_max": signals["rps_doc_frac_lines_end_with_ellipsis"][0][2] < self.filters["lines_end_with_ellipsis_max"],
            "bullet_point_lines_max_ratio": sum(ln[2] for ln in signals["rps_lines_start_with_bulletpoint"]) / signals["ccnet_nlines"][0][2] <= self.filters["bullet_point_lines_max_ratio"],
            "no_alph_words_max": signals["rps_doc_frac_no_alph_words"][0][2] < self.filters["no_alph_words_max"],
            "stop_word_fraction_min": signals["rps_doc_stop_word_fraction"][0][2] >= self.filters["stop_word_fraction_min"],
            "symbol_to_word_ratio_max": signals["rps_doc_symbol_to_word_ratio"][0][2] < self.filters["symbol_to_word_ratio_max"],
            "word_count_range": self.in_range(signals["rps_doc_word_count"][0][2], self.filters["word_count_range"]),
            "curly_bracket_max": signals["rps_doc_curly_bracket"][0][2] < self.filters["curly_bracket_max"],
            "lorem_ipsum_max": signals["rps_doc_lorem_ipsum"][0][2] < self.filters["lorem_ipsum_max"],
            #"books_importance_range": self.in_range(signals["rps_doc_books_importance"][0][2], self.filters["books_importance_range"]),
            #"openwebtext_importance_min": signals["rps_doc_openwebtext_importance"][0][2] or float("-inf") > self.filters["openwebtext_importance_min"],
            "unigram_entropy_range": self.in_range(signals["rps_doc_unigram_entropy"][0][2], self.filters["unigram_entropy_range"]),
            "top_2_gram_frac_max": signals["rps_doc_frac_chars_top_2gram"][0][2] <= self.filters["top_2_gram_frac_max"],
            "top_3_gram_frac_max": signals["rps_doc_frac_chars_top_3gram"][0][2] <= self.filters["top_3_gram_frac_max"],
            "top_4_gram_frac_max": signals["rps_doc_frac_chars_top_4gram"][0][2] <= self.filters["top_4_gram_frac_max"],
            "dupe_5_gram_frac_max": signals["rps_doc_frac_chars_dupe_5grams"][0][2] <= self.filters["dupe_5_gram_frac_max"],
            "dupe_6_gram_frac_max": signals["rps_doc_frac_chars_dupe_6grams"][0][2] <= self.filters["dupe_6_gram_frac_max"],
            "dupe_7_gram_frac_max": signals["rps_doc_frac_chars_dupe_7grams"][0][2] <= self.filters["dupe_7_gram_frac_max"],
            "dupe_8_gram_frac_max": signals["rps_doc_frac_chars_dupe_8grams"][0][2] <= self.filters["dupe_8_gram_frac_max"],
            "dupe_9_gram_frac_max": signals["rps_doc_frac_chars_dupe_9grams"][0][2] <= self.filters["dupe_9_gram_frac_max"],
            "dupe_10_gram_frac_max": signals["rps_doc_frac_chars_dupe_10grams"][0][2] <= self.filters["dupe_10_gram_frac_max"],
            "allow_dupe": not signals["is_duplicate"] or self.filters["allow_dupe"],
            "language": self.filters["language"] is None or self.filters["language"] == meta["language"]
        }

        # Increment the counter for each failed rule
        for rule, passed in rule_checks.items():
            if not passed:
                self.fail_counters[rule] += 1

        # Return True only if all rules are passed
        return all(rule_checks.values())


# Function to load dataset and apply the DocumentFilter
def create_filtered_dataset(language=None):
    # Load the RedPajama-Data-V2 dataset from Hugging Face
    dataset = load_dataset(
        "togethercomputer/RedPajama-Data-V2",
        name="sample-10B",
        partition="head_middle",
        languages=[language] if language else [],
        split="train",
        trust_remote_code=True
    )
    import pdb;pdb.set_trace()

    # Create a DocumentFilter instance (ignoring stop words check)
    doc_filter = DocumentFilter(language=language)

    # Apply the filter to the dataset using a lambda function
    filtered_dataset = dataset.filter(lambda example: doc_filter(example))

    # print stats
    for x in sorted(doc_filter.fail_counters.items(), key=lambda x: x[1]):
        print(x)

     # Return the filtered dataset
    return filtered_dataset.rename_column("raw_content", "text")


# Example usage
create_filtered_dataset().push_to_hub("distily/filtered_redpajama_multilingual")
create_filtered_dataset(language="en").push_to_hub("distily/filtered_redpajama_en")
