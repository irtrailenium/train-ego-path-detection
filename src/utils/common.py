import logging
import random

import numpy as np
import torch
from torchvision.transforms import v2 as transforms

to_scaled_tensor = transforms.Compose(
    [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
)  # [0, 255] PIL.Image or numpy.ndarray to [0, 1] torchvision Image (torch.Tensor)


def split_dataset(indices, proportions):
    """Splits the dataset indices into training, validation and test sets.

    Args:
        indices (list): List of indices of the dataset.
        proportions (tuple): Tuple containing the proportions of the training, validation and test sets (e.g. (0.8, 0.1, 0.1)).

    Returns:
        tuple: Tuple containing the training, validation and test sets indices.
    """
    train_prop, val_prop, test_prop = proportions
    train_indices = indices[: int(train_prop * len(indices))]
    val_indices = indices[
        int(train_prop * len(indices)) : int((train_prop + val_prop) * len(indices))
    ]
    test_indices = indices[
        int((train_prop + val_prop) * len(indices)) : int(
            (train_prop + val_prop + test_prop) * len(indices)
        )
    ]
    return train_indices, val_indices, test_indices


def simple_logger(name, level, terminator="\n"):
    """Creates a simple logger to print messages to the console with no additional information.

    Args:
        name (str): Identifier of the logger.
        level (str): Level of the logger. Only messages with this level or higher will be printed.
        terminator (str, optional): String to append to the end of each message. Defaults to "\n".

    Returns:
        logging.Logger: Simple logger.
    """
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logger = logging.getLogger(name)
    logger.setLevel(levels[level])
    handler = logging.StreamHandler()
    handler.setLevel(levels[level])
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.terminator = terminator
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_worker_seeds(worker_id):
    seed = torch.initial_seed() % 2**32  # clamp to 32-bit
    random.seed(seed)
    np.random.seed(seed)
