from . import args, benchmark, cli, distillation_trainer, objectives, metrics
from . import tinyllama_bitnet_utils  # TODO: Remove, replace with a good bitnet library

import importlib.metadata
__version__ = importlib.metadata.version("distily")
