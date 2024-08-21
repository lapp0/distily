from . import (  # noqa
    args,
    bitnet_utils,
    data,
    distillation_trainer,
    metrics,
    models,
    objectives,
    run,
)


import importlib.metadata
__version__ = importlib.metadata.version("distily")
