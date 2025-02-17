import torch
from torch import nn
import matplotlib
import numpy as np
import random
import pandas as pd
from skimage.transform import resize
import gc
import subprocess
import os
import dill

def setup(config):
    set_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    gc.collect()
    torch.cuda.empty_cache()

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

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode('utf-8')
    except Exception as e:
        print(f"Error retrieving Git information: {e}")
        commit = "unknown"
        branch = "unknown"
    return commit, branch

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
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def save_all(model, avg_train_loss, avg_valid_loss, time_per_epoch, config):
    # Save everything
    print("Saving to", config.parent_dir)
    if os.path.exists(os.path.join(config.parent_dir, 'model.pt')):
        if input("Path exists. Overwrite? (y/n)") == 'y':
            torch.save(model, os.path.join(config.parent_dir, 'model.pt'), pickle_module=dill)
            avg_train_loss_arr = np.array(avg_train_loss)
            avg_valid_loss_arr = np.array(avg_valid_loss)
            time_arr = np.array(time_per_epoch)
            np.save(os.path.join(config.parent_dir, 'train_loss.npy'), avg_train_loss_arr)
            np.save(os.path.join(config.parent_dir, 'valid_loss.npy'), avg_valid_loss_arr)
            np.save(os.path.join(config.parent_dir, 'time.npy'), time_arr)
            torch.save(config, os.path.join(config.parent_dir, 'config.pt'))
            print("Saved successfully to", config.parent_dir)
        else:
            print("Saving failed.")
    else:
        torch.save(model, os.path.join(config.parent_dir, 'model.pt'), pickle_module=dill)
        avg_train_loss_arr = np.array(avg_train_loss)
        avg_valid_loss_arr = np.array(avg_valid_loss)
        time_arr = np.array(time_per_epoch)
        np.save(os.path.join(config.parent_dir, 'train_loss.npy'), avg_train_loss_arr)
        np.save(os.path.join(config.parent_dir, 'valid_loss.npy'), avg_valid_loss_arr)
        np.save(os.path.join(config.parent_dir, 'time.npy'), time_arr)
        torch.save(config, os.path.join(config.parent_dir, 'config.pt'))
        print("Saved successfully to", config.parent_dir)

def _to_sequence(inp, inv=False, orig_shape=None):
    if not inv:
        nb, nx, nz = inp.shape
        out = inp.transpose(-1, -2) # batch, z, x
        out = out.reshape(-1, out.shape[-1]).unsqueeze(1)  # flatten images into sequences
        
        return out, (nb, nx, nz)
    
    elif inv and orig_shape is not None:
        out = inp.squeeze(1).reshape(orig_shape[0], -1, orig_shape[1])
        out = out.transpose(-1, -2) # batch, x, z

        return out
    
def _to_sequence2(inp, inv=False, orig_shape=None):
    if not inv:
        nb, nx, nz = inp.shape
        out = inp.transpose(-1, -2).reshape(nb, -1) # batch, L
        out = out.transpose(0, 1)  # L, batch
        
        return out, (nb, nx, nz)
    
    elif inv and orig_shape is not None:
        out = inp.transpose(0, 1) # batch, L
        out = out.reshape(-1, orig_shape[2], orig_shape[1]).transpose(-1, -2)
        
        return out
    
def get_lr(optim):
    for param_group in optim.param_groups:
        return param_group['lr']
    
def PSNR(x, xinv):
    return 10 * torch.log10(len(xinv) * torch.max(xinv) ** 2 / torch.linalg.norm(x - xinv) ** 2)

def set_dropout_prob(model, p=0.1):
    for idx, m in enumerate(model.named_modules()): 
        component = m[1]
        if isinstance(component, nn.Dropout):
            component.p = p