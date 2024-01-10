import torch
import matplotlib
import numpy as np
import random
import torch.nn.functional as F
from scipy import signal
from scipy.ndimage import gaussian_filter
import pandas as pd

def set_mpl_params():
    """Set matplotlib parameters for plotting."""

    params = {
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'figure.dpi' : 300,
    'axes.labelsize':14,  # fontsize for x and y labels (was 10)
    'axes.titlesize':14,
    'axes.titleweight': 'bold',
    'legend.fontsize': 14,  # was 10
    'xtick.labelsize':12,
    'ytick.labelsize':12,
    'font.family': 'serif',
    'font.size': 12,
    'figure.titleweight': 'bold',
    'figure.titlesize': 14 
}
    matplotlib.rcParams.update(params)

def seed_worker(worker_id):
    """Fix seed of the workers (used in PyTorch dataloader)."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    """Fix seed of random, PyTorch, and numpy random state.

    Args:
        seed (int): seed to be set.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def count_parameters(model):
    """Count the number of trainable parameters of a model.

    Args:
        model (:class:`~torch.nn.Module`): PyTorch model of which the number of parameters to be counted.

    Returns:
        int: Total number of trainable parameters.
    """
    
    table = pd.DataFrame(columns=['Name', 'Parameter'])
    total_params = 0
    i = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
        table.loc[i] = [name] + [param]
        i += 1
    display(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params