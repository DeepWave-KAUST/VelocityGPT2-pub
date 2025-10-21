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
from typing import OrderedDict
import wandb
import warnings
import math
from pytorch_msssim import ssim
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor

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

def save_all(model, avg_train_loss, avg_valid_loss, time_per_epoch, config, optim, scheduler):
    # Save everything
    print("Saving to", config.parent_dir)
    if os.path.exists(os.path.join(config.parent_dir, 'model.pt')) and config.cont_dir is None:
        if input("Path exists. Overwrite? (y/n)") == 'y':
            torch.save(model, os.path.join(config.parent_dir, 'model.pt'), pickle_module=dill)
            if optim is not None:
                torch.save(optim.state_dict(), os.path.join(config.parent_dir, 'optim.pt'))
            if scheduler is not None:
                torch.save(scheduler.state_dict(), os.path.join(config.parent_dir, 'scheduler.pt'))
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
        if optim is not None:
            torch.save(optim.state_dict(), os.path.join(config.parent_dir, 'optim.pt'))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(config.parent_dir, 'scheduler.pt'))
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

def load_from_pt_or_checkpoint(path, key):
    if os.path.exists(os.path.join(path, key+'.pt')):
        obj = torch.load(os.path.join(path, key+'.pt'), 
                         pickle_module=dill if key == 'model' else None, 
                         map_location='cpu')
        if isinstance(obj, OrderedDict) or isinstance(obj, dict):
            print(f"Loading {key} from {path}")
            return obj
        else:
            print(f"Loading {key} from {path}")
            return obj.state_dict()
    elif any(['checkpoint' in x for x in os.listdir(path)]):
        obj = torch.load(os.path.join(path, [x for x in os.listdir(path) if 'checkpoint' in x][0]), 
                         map_location='cpu')
        print(f"Loading {key} from {path} checkpoint")
        return obj[key]
    else:
        raise FileNotFoundError(f"Checkpoint or {key}.pt not found in {path}")
    
def get_previous_epoch_count(config):
    """Determine the previous epoch count when resuming training.
    
    Args:
        config: Configuration object containing training parameters.
            Required fields when resuming:
                - cont_dir: Directory containing previous training data
                - Optional: wandb_log, wandb_id for Weights & Biases log information
    
    Returns:
        int: Number of epochs already completed (0 if cannot be determined)
    """
    if not hasattr(config, 'cont_dir') or config.cont_dir is None:
        return 0
    
    # Method 1: Get epoch from wandb (highest priority)
    if hasattr(config, 'wandb_log') and config.wandb_log and hasattr(config, 'wandb_id') and config.wandb_id:
        try:
            api = wandb.Api()
            run = api.run('caezario/ElasticGPT/'+config.wandb_id)
            if 'epoch' in run.summary:
                print(f"Resuming from epoch {run.summary['epoch']+1} (from W&B)")
                return run.summary['epoch'] + 1
            elif 'epoch' in run.history():
                history = run.scan_history(keys=['epoch'])
                if history:
                    epoch = max([h['epoch'] for h in history])
                    print(f"Resuming from epoch {epoch} (from W&B history)")
                    return epoch + 1
            else:
                warnings.warn("W&B run found but no epoch information available")
        except Exception as e:
            warnings.warn(f"Failed to get epoch from W&B: {e}")
    
    # Method 2: Check if train_loss.npy exists and use its length
    train_loss_path = os.path.join(config.cont_dir, 'train_loss.npy')
    if os.path.exists(train_loss_path):
        try:
            losses = np.load(train_loss_path)
            prev_epochs = len(losses)
            print(f"Resuming from epoch {prev_epochs} (from train_loss.npy)")
            return prev_epochs
        except Exception as e:
            warnings.warn(f"Failed to load train_loss.npy: {e}")
    
    # Method 3: Try to extract information from optimizer state
    optim_state_path = os.path.join(config.cont_dir, 'optimizer.pt')
    if os.path.exists(optim_state_path):
        try:
            optim_state = torch.load(optim_state_path)
            if 'state' in optim_state and len(optim_state['state']) > 0:
                # Some optimizers store step count in their state
                first_param_state = next(iter(optim_state['state'].values()))
                if 'step' in first_param_state:
                    step_count = first_param_state['step']
                    
                    # Need batch size to convert steps to epochs
                    config_path = os.path.join(config.cont_dir, 'config.pt')
                    if os.path.exists(config_path):
                        try:
                            prev_config = torch.load(config_path, pickle_module=dill)
                            if hasattr(prev_config, 'batch_size') and hasattr(prev_config, 'train_size'):
                                steps_per_epoch = prev_config.train_size // prev_config.batch_size
                                if steps_per_epoch > 0:
                                    prev_epochs = step_count // steps_per_epoch
                                    print(f"Resuming from epoch {prev_epochs} (estimated from optimizer state)")
                                    return prev_epochs
                        except Exception as e:
                            warnings.warn(f"Failed to load previous config: {e}")
        except Exception as e:
            warnings.warn(f"Failed to analyze optimizer state: {e}")
    
    # Method 4: Check for epoch information in checkpoint files
    checkpoint_files = [f for f in os.listdir(config.cont_dir) if 'checkpoint' in f]
    if checkpoint_files:
        try:
            checkpoint = torch.load(os.path.join(config.cont_dir, checkpoint_files[0]))
            if 'epoch' in checkpoint:
                print(f"Resuming from epoch {checkpoint['epoch']+1} (from checkpoint file)")
                return checkpoint['epoch'] + 1
        except Exception as e:
            warnings.warn(f"Failed to get epoch from checkpoint: {e}")
    
    # Method 5: If config.pt exists, check if it has an 'epoch' attribute
    config_path = os.path.join(config.cont_dir, 'config.pt')
    if os.path.exists(config_path):
        try:
            prev_config = torch.load(config_path, pickle_module=dill)
            if hasattr(prev_config, 'last_epoch'):
                print(f"Resuming from epoch {prev_config.last_epoch} (from config.pt)")
                return prev_config.last_epoch
        except Exception as e:
            warnings.warn(f"Failed to check config.pt for epoch information: {e}")
            
    print("Could not determine previous epoch count. Starting from epoch 0.")
    return 0

def calc_teacher_forcing_ratio(epoch, config):
    
    if config.sampling_type == "teacher_forcing":
        return 1.0
    elif config.sampling_type == "scheduled":  # scheduled sampling
        if config.scheduled_sampling_decay == "exp":
            scheduled_ratio = config.scheduled_sampling_k ** epoch
        elif config.scheduled_sampling_decay == "sigmoid":
            if epoch / config.scheduled_sampling_k > 700:
                scheduled_ratio = 0
            else:
                scheduled_ratio = config.scheduled_sampling_k / (
                        config.scheduled_sampling_k
                        + math.exp(epoch / config.scheduled_sampling_k)
                        )
        else:  # linear 
            scheduled_ratio = config.scheduled_sampling_k - \
                                config.scheduled_sampling_c * epoch
        scheduled_ratio = max(config.scheduled_sampling_limit, scheduled_ratio)
        return scheduled_ratio
    else:  # always sample from the model predictions
        return 0.0
    
def count_layers_with_clustering(velocity_model, velocity_threshold=1e-3, spatial_weight=1.0):
    """
    Count the number of continuous layers in a velocity model using clustering.

    Args:
        velocity_model (np.ndarray): 2D array of shape (nz, nx) representing the velocity model.
        velocity_threshold (float): Threshold for velocity similarity.
        spatial_weight (float): Weight to balance the contribution of spatial continuity.

    Returns:
        int: Number of detected continuous layers.
    """
    nz, nx = velocity_model.shape

    # Flatten the model and create spatial coordinates
    velocities = velocity_model.flatten()  # Already NumPy array, no need for .numpy()
    z_coords, x_coords = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    z_coords, x_coords = z_coords.flatten(), x_coords.flatten()

    # Scale spatial coordinates to balance with velocity
    z_coords = z_coords * spatial_weight
    x_coords = x_coords * spatial_weight

    # Combine velocity and spatial coordinates into a single feature space
    features = np.stack([velocities, z_coords, x_coords], axis=1)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=velocity_threshold, min_samples=1, metric="euclidean")
    labels = dbscan.fit_predict(features)

    # Count the unique clusters (layers)
    num_layers = len(np.unique(labels))
    return num_layers


def batch_count_layers(samples, velocity_threshold, spatial_weight):
    """
    Count layers for multiple samples in parallel while maintaining order.

    Args:
        samples (torch.Tensor): Batch of velocity models (PyTorch tensors).
        velocity_threshold (float): Clustering velocity threshold.
        spatial_weight (float): Clustering spatial weight.

    Returns:
        list: Layer counts for each sample (ordered by input index).
    """
    # Ensure the samples are in NumPy format for compatibility with multiprocessing
    samples = [sample.cpu().numpy() for sample in samples]  # Explicit conversion from PyTorch tensors

    with ProcessPoolExecutor() as executor:
        # Use enumerate to keep track of indices for ordering
        futures = [
            executor.submit(count_layers_with_clustering, sample, velocity_threshold, spatial_weight)
            for sample in samples
        ]

        # Collect results in order of submission
        results = [future.result() for future in futures]

    return results
    
def evaluate_generated_models(velocity_models, eval_n_layers=True, velocity_threshold=1e-3, spatial_weight=1.0, prefix=""):
    """
    Evaluate generated velocity models against the ground truth using clustering for layer detection.

    Args:
        velocity_models (torch.Tensor): Tensor of shape (num_samples, num_models+2, nx, nz).
                                        The first index is the input, the last index is the ground truth,
                                        and the remaining indices are the generated samples.
        velocity_threshold (float): The threshold for velocity similarity in layer detection.
        spatial_weight (float): The weight to balance spatial continuity in clustering.

    Returns:
        dict: A dictionary containing RMSE, MAE, SSIM, and layer similarity metrics.
    """
    # Extract the generated samples and ground truth
    generated_samples = velocity_models[:, 1:-1, :, :]  # Exclude input and ground truth
    ground_truth = velocity_models[:, -1, :, :]  # Ground truth is the last index

    # Calculate RMSE
    rmse = torch.sqrt(((generated_samples - ground_truth.unsqueeze(1)) ** 2).mean(dim=(2, 3))).mean(dim=(0, 1))

    # Calculate MAE
    mae = (generated_samples - ground_truth.unsqueeze(1)).abs().mean(dim=(2, 3)).mean(dim=(0, 1))

    # Calculate SSIM
    ssim_values = []
    for i in range(generated_samples.shape[1]):  # Loop over generated models
        ssim_vals = [ssim(gen[None, None, ...], gt[None, None, ...], data_range=4500.0)
                     for gen, gt in zip(generated_samples[:, i, :, :], ground_truth)]
        ssim_values.append(torch.stack(ssim_vals).mean())
    avg_ssim = torch.tensor(ssim_values).mean()

    # Layer counting using clustering
    if eval_n_layers:
        flattened_generated = generated_samples.reshape(-1, *generated_samples.shape[2:])
        flattened_ground_truth = ground_truth

        generated_layers = torch.tensor(
            batch_count_layers(flattened_generated, velocity_threshold, spatial_weight)
        ).reshape(generated_samples.shape[:2])

        ground_truth_layers = torch.tensor(
            batch_count_layers(flattened_ground_truth, velocity_threshold, spatial_weight)
        )

        # Calculate average absolute layer difference
        layer_diff = (generated_layers - ground_truth_layers.unsqueeze(1)).abs().float().mean()
    else:
        layer_diff = torch.tensor(0)
        generated_layers = torch.tensor(0)
        ground_truth_layers = torch.tensor(0)

    # Prepare the results
    results = {
        prefix+"RMSE": rmse.item(),
        prefix+"MAE": mae.item(),
        prefix+"SSIM": avg_ssim.item(),
        prefix+"Layer Difference": layer_diff.item(),
        prefix+"Generated Layers": generated_layers,
        prefix+"Ground Truth Layers": ground_truth_layers
    }
    return results