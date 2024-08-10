from itertools import product
import re
import gc
import torch
import distily


def get_run_name(run_kwargs):
    normalize = lambda s: re.sub(r'[^a-zA-Z0-9]', '_', str(s))
    # Create a sorted list of normalized key-value pairs joined by underscores
    return ", ".join([
        f"{normalize(k)}={normalize(v)}"
        for k, v in sorted(run_kwargs.items())
    ])


def run(product_kwargs, **kwargs):
    """
    Benchmark the training process by exploring permutations of hyperparameters.

    This function takes a set of hyperparameters as keyword arguments, where each
    value is a list of possible options for that hyperparameter. It generates all
    possible combinations of the hyperparameters and calls the training function
    for each combination.

    Parameters:
    **kwargs: Arbitrary keyword arguments where each value is a list of possible
              values for the corresponding hyperparameter.

    Raises:
    ValueError: If any of the provided keyword arguments is not a list.

    Example:
    benchmark(learning_rate=[4e-5, 4e-4], optim=["lion", "adamw"])
    """
    for key, value in product_kwargs.items():
        if not isinstance(value, list):
            raise ValueError(f"The value for '{key}' must be a list.")

    # Get all combinations of the items in the lists
    keys = kwargs.keys()
    values_product = product(*kwargs.values())

    # log params
    print("Training Parameters")
    print("\n".join([
        str(dict(zip(keys, vals)))
        for vals in values_product
    ]))

    for values in values_product:
        product_args = dict(zip(keys, values))
        product_args["run_name"] = get_run_name(product_args)
        current_args = {
            **product_args,
            **kwargs
        }
        training_args, student_model_args, teacher_model_args, dataset_args = distily.args.parser.parse_dict(
            current_args,
            allow_extra_keys=True
        )

        try:
            # TODO: do_train should return training results
            res = distily.cli.do_train(training_args, student_model_args, teacher_model_args, dataset_args)
        except Exception as e:
            print(f"FAILED FOR {current_args}")
            print(e)

        print(f"SUCCESS FOR {current_args}")

        # cleanup
        gc.collect()
        torch.cuda.empty_cache()
