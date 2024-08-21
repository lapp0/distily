import traceback
import os
import re
import gc
import torch

import distily


def benchmark(params=None, **kwargs):
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
    def get_run_name(run_kwargs):
        normalize = lambda s: re.sub(r'[^A-Za-z0-9_\-\.()]', '_', s if isinstance(s, str) else repr(s))
        # Create a sorted list of normalized key-value pairs joined by underscores
        return ", ".join([
            f"{normalize(k)}={normalize(v)}"
            for k, v in sorted(run_kwargs.items())
        ])[:200]

    assert params is not None

    # log params
    print("Training Parameters")
    print("\n".join(map(str, params)))

    for values in params:
        product_args = dict(values)
        run_name = get_run_name(product_args)
        print(run_name)
        current_args = {
            **product_args,
            **kwargs
        }
        current_args["logging_dir"] = os.path.join(current_args["output_dir"], "logs", run_name)

        completion_flag = os.path.join(current_args["logging_dir"], "completed.flag")
        if os.path.exists(completion_flag):
            print(f"Run '{run_name}' has already been completed. Skipping...")
            continue

        parsed_args_tuple = distily.args.parser.parse_dict(
            current_args,
            allow_extra_keys=True
        )

        try:
            # TODO: train should return training results
            res = train(*parsed_args_tuple)

            open(completion_flag, 'a').close()  # write completion flag

        except Exception as e:
            print(f"FAILED FOR {current_args}")
            print(e)
            traceback.print_exc()
            continue
        print(f"SUCCESS FOR {current_args}")

        # cleanup
        gc.collect()
        torch.cuda.empty_cache()


def train(training_args, distillation_objective_args, student_model_args, teacher_model_args, dataset_args):
    trainer = distily.distillation_trainer.DistillationTrainer.from_args(
        training_args, distillation_objective_args, student_model_args, teacher_model_args, dataset_args
    )
    trainer.train()
    if training_args.push_to_hub:
        trainer.push_to_hub()


def train_entry():
    train(*distily.args.get_args())
