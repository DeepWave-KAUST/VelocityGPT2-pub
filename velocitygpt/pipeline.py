import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from radam import RAdam
import sys
import pandas as pd
import matplotlib
from scipy import signal
from natsort import natsorted
from scipy.io import loadmat
import math
import gc
import wandb
from pynvml.smi import nvidia_smi
from pytorch_msssim import ssim
import copy
import pytorch_warmup as pw
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from torchinfo import summary
import multiprocessing
import pylops
from pylops.utils.wavelets import ricker

from .utils import *
from .datasets import *
from .datasets import _get_dip
from .losses import *
from .models import *
from .quantizer import *

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)
torch.cuda.empty_cache()


def load_and_prep(config):
    device = config.device
    if isinstance(config.dataset_path, list):
        train_data = []
        test_data = []
        for i, path in enumerate(config.dataset_path):
            if i == 0:
                train_data = torch.load(os.path.join(path, 'train_data.pt'))
                test_data = torch.load(os.path.join(path, 'test_data.pt'))
                if config.classify or config.cls_token:
                    for data in [train_data, test_data]:
                        data.data['cls'] = i * torch.ones(data.data['input'].shape[0]).long()
            else:
                train_data_ = torch.load(os.path.join(path, 'train_data.pt'))
                test_data_ = torch.load(os.path.join(path, 'test_data.pt'))
                if config.classify or config.cls_token:
                    for data in [train_data_, test_data_]:
                        data.data['cls'] = i * torch.ones(data.data['input'].shape[0]).long()
                for data, data_ in zip([train_data, test_data], [train_data_, test_data_]):
                    for key in data.data.keys():
                        data.data[key] = torch.cat((data.data[key], data_.data[key]), dim=0)
    else:
        train_data = torch.load(os.path.join(config.dataset_path, 'train_data.pt'))
        test_data = torch.load(os.path.join(config.dataset_path, 'test_data.pt'))
        
    if config.shuffle or config.train_prop:
        for i, key in enumerate(train_data.data.keys()):
            all_data = torch.cat((train_data.data[key], test_data.data[key]), dim=0)
            if key == 'cls' and config.cls_token:
                if config.num_classes != all_data.max().item():
                    ns = len(all_data)
                    nw = round(ns / all_data.max().item())
                    ng = int(np.ceil(ns / config.num_classes / nw))
                    new_nw = nw * ng
                    all_data = torch.arange(config.num_classes).repeat_interleave(new_nw)
                    all_data = all_data[:ns]
            if i == 0:
                shuffle_idx = torch.randperm(len(all_data))
            if config.shuffle:
                all_data = all_data[shuffle_idx]
            if config.train_prop:
                train_len = round(config.train_prop*len(all_data))
                train_data.data[key] = all_data[:train_len]
                test_data.data[key] = all_data[train_len:]              
            del all_data
            
    scaler1 = []
    pad = None
    
    for i, data in enumerate([train_data, test_data]):
        if config.prop is not None:
            data_len = max(int(config.prop * len(data)), 1) # Min no of samples = 1
            for key in data.data.keys():
                data.data[key] = data.data[key][:data_len]
        if config.dataset_type == "syn1":
            for key in data.data.keys():
                if key in ['input', 'label']:
                    data.data[key] = data.data[key][:, 3:-3, 3:-3]
        # Delete dip if not needed
        if not config.use_dip:
            try:
                del data.data["dip"]
                del data.data["dip_seq"]
            except:
                pass
        # Smooth
        if config.smooth_class and config.smooth: # Apply gaussian smoothing
            for cls, s in zip(config.smooth_class, config.smooth):
                smoothed = []
                for x, y in zip(data.data['input'], data.data['cls']):
                    if y.item() in cls:
                        smoothed.append(torch.tensor(gaussian_filter(x.numpy(), [s, s])).float())
                smoothed = torch.stack(smoothed)
                if config.use_dip:
                    dip_seq, dip = _get_dip(smoothed.numpy(), config.dip_bins, config.patch_size, True)
                for key in data.data.keys():
                    if key == 'cls':
                        new_cls = torch.ones(smoothed.shape[0]).long() * (torch.max(data.data['cls']) + 1)
                        data.data[key] = torch.cat((data.data[key], new_cls), dim=0)
                    elif key == "dip_seq":
                        data.data[key] = torch.cat((data.data[key], dip_seq), dim=0)
                    elif key == "dip":
                        data.data[key] = torch.cat((data.data[key], dip), dim=0)
                    else:
                        data.data[key] = torch.cat((data.data[key], smoothed), dim=0)
                
        if config.image_size != config.orig_image_size:
            if config.pad_input:
                # Pad
                pad0 = int((config.image_size[1] - config.orig_image_size[1]) // 2)
                pad1 = int((config.image_size[1] - config.orig_image_size[1]) - pad0)
                pad2 = int((config.image_size[0] - config.orig_image_size[0]) // 2)
                pad3 = int((config.image_size[0] - config.orig_image_size[0]) - pad2)
                pad = [pad0, pad1, pad2, pad3]
                for key in data.data.keys():
                    if key in ['input', 'label']:
                        data.data[key] = pad_input(data.data[key], pad)      
            else:
                # Resize
                for key in data.data.keys():
                    if key in ['input', 'label']:
                        resized_images = []
                        for j in range(len(data.data[key])):
                            resized_images.append(torch.tensor(resize_image2(data.data[key][j].numpy(), 
                                                                             config.image_size)))
                        data.data[key] = torch.stack(resized_images)
        # Compress
        if config.compress_class and config.compress_ratio:
            for cls, r in zip(config.compress_class, config.compress_ratio):
                compressed = []
                comp_size = (config.image_size[0], int(config.image_size[1]/r))
                for x, y in zip(data.data['input'], data.data['cls']):
                    if y.item() in cls:
                        compressed.append(torch.tensor(resize_image2(x.numpy(), comp_size)).float())
                compressed = torch.stack(compressed)
                if config.compress_shuffle:
                    set_seed(config.seed)
                    shuffled_idx = torch.randperm(len(compressed))
                    compressed = compressed[shuffled_idx]
                compressed = compressed.reshape(-1, r, config.image_size[0], comp_size[1])
                compressed = compressed.permute(0, 2, 1, 3)
                compressed = compressed.reshape(-1, config.image_size[0], config.image_size[1])
                if config.use_dip:
                    dip_seq, dip = _get_dip(compressed.numpy(), config.dip_bins, config.patch_size, True)
                for key in data.data.keys():
                    if key == 'cls':
                        new_cls = torch.ones(compressed.shape[0]).long() * (torch.max(data.data['cls']) + 1)
                        data.data[key] = torch.cat((data.data[key], new_cls), dim=0)
                    elif key == "dip_seq":
                        data.data[key] = torch.cat((data.data[key], dip_seq), dim=0)
                    elif key == "dip":
                        data.data[key] = torch.cat((data.data[key], dip), dim=0)
                    else:
                        data.data[key] = torch.cat((data.data[key], compressed), dim=0)
                        
        # Reflectivity / Image
        if config.vqvae_refl_dir is not None or config.training_stage == "vqvae-training-refl":
            if 'refl' not in data.data.keys():
                AI = make_AI(data.data['input'])
                if config.input_type == "refl":
                    inp = make_refl(AI)
                elif config.input_type == "img":
                    t0 = np.arange(config.nt0) * config.dt0
                    wav, twav, wavc = ricker(t0[: config.ntwav // 2 + 1], config.freq)
                    inp = make_post(AI.numpy(), wav)
                data.data['refl'] = inp.clone()
                if config.training_stage == "vqvae-training-refl":
                    data.data['input'] = inp.clone()
                    data.data['label'] = inp.clone()
            elif 'refl' in data.data.keys() and config.training_stage == "vqvae-training-refl":
                data.data['input'] = data.data['refl'].clone()
                data.data['label'] = data.data['refl'].clone()
                        
        # Flip
        if config.aug_flip:
            for key in data.data.keys():
                if key in ['input', 'label', 'refl']:
                    data.data[key] = torch.cat((data.data[key], 
                                                torch.flip(data.data[key], dims=(1,))), dim=0)
                elif key == 'dip_seq':
                    dip_seq_flip = data.data[key].clone()
                    dip_seq_flip = dip_seq_flip.reshape(-1, config.latent_dim, config.latent_dim)
                    dip_seq_flip = torch.flip(dip_seq_flip, dims=(2,))
                    dip_seq_flip = dip_seq_flip.reshape(dip_seq_flip.shape[0], -1)
                    data.data[key] = torch.cat((data.data[key], dip_seq_flip), dim=0)
                else:
                    data.data[key] = torch.cat((data.data[key], data.data[key]), dim=0)
                    
#         # Add dip info
#         if config.use_dip:
#             bins = np.array(config.dip_bins)
#             dip = []
#             for j in tqdm(range(len(data.data['input']))):
#                 coh, pp, res = SeisPWD(data.data['input'][j].numpy(), w1=5, w2=5)
#                 dip_ = np.digitize(pp, bins)
#                 patch_size = np.array(config.image_size) // config.latent_dim
#                 dip_ = patchify(torch.tensor(dip_), patch_size, patch_size)
#                 dip_ = dip_.mode(-1).values.mode(-1).values
#                 dip.append(dip_)
#             dip = torch.stack(dip, dim=0)
#             data.data['dip'] = dip
                  
        if config.norm_mode == "independent":
            scaler1.append(torch.abs(data.data['input']).max(-1).values.max(-1).values)
        elif config.norm_mode == "set":
            scaler1.append(torch.ones(data.data['input'].shape[0]) * data.data['input'].max())
        elif config.norm_mode == "manual":
            scaler1.append(torch.ones(data.data['input'].shape[0]) * config.norm_const)
        data.data['input'] = (data.data['input'] / scaler1[i][:, None, None] - config.scaler3) * config.scaler2
        data.data['label'] = (data.data['label'] / scaler1[i][:, None, None] - config.scaler3) * config.scaler2
        scaler1[i] = scaler1[i].to(config.device) 
        if config.use_dip:
            data.data['dip_seq'] -= 1
        if config.vqvae_refl_dir is not None:
            scaler_refl = torch.abs(data.data['refl']).max(-1).values.max(-1).values[:, None, None]
            data.data['refl'] = data.data['refl'] / scaler_refl
        
    return train_data, test_data, scaler1, pad

def build_dataloader(config, train_data, test_data):
    batch_size = config.batch_size

    g = torch.Generator()
    g.manual_seed(config.seed)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker, num_workers=4, persistent_workers=True, prefetch_factor=8)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, prefetch_factor=8)
    
    return train_dataloader, test_dataloader

def load_model(config, model_type="vel"):
    device = config.device
    if model_type == "vel":
        model = torch.load(os.path.join(config.vqvae_dir, "model.pt"), map_location='cpu')
    elif model_type == "refl":
        model = torch.load(os.path.join(config.vqvae_refl_dir, "model.pt"), map_location='cpu')
    model.eval()
    
    return model.to(config.device)

def build_model(config):
    device = config.device
    set_seed(config.seed)
    if "vqvae" in config.training_stage:
        if config.vq_type == "vqvae":
            model = VectorQuantizedVAE(config)
        elif config.vq_type == "vqvae2":
            model = VQVAE(config)
            
        print(summary(model.to('cuda:0'), 
                    input_size=(config.batch_size, 1, config.image_size[0], config.image_size[1]), 
                    device=config.device))
        
    else:
        model = GPT2(config)
        print(model)
        latents = torch.randint(high=config.vocab_size, size=(config.max_length, config.batch_size))
        cls = torch.randint(high=config.num_classes, size=(config.batch_size,))
        if config.well_cond_prob > 0:
            well_pos = torch.randint(high=config.image_size[0], size=(config.batch_size,))
            latents_well = torch.randint(high=config.vocab_size, size=(config.latent_dim, config.batch_size))
            input_data = [latents, cls, well_pos, latents_well]
        else:
            well_pos = None
            latents_well = None
            input_data = [latents, cls, well_pos, latents_well]
        if config.use_dip:
            dip = torch.randint(high=len(config.dip_bins), size=(config.batch_size, config.max_length))
            input_data.append(dip)
        else:
            input_data.append(None)
        if config.vqvae_refl_dir is not None:
            refl = torch.randint(high=config.refl_vocab_size, size=(config.max_length, config.batch_size))
            input_data.append(refl)
        else:
            input_data.append(None)
        if config.add_dip_to_well:
            dip_well = torch.randint(high=len(config.dip_bins), size=(config.batch_size, config.latent_dim))
            input_data.append(dip_well)
        else:
            input_data.append(None)
        print(summary(model.to(config.device), 
                    input_data=input_data, 
                    device=config.device))
    
    return model.to(config.device)

# Optimizer
def build_optimizer(config, model):
    if config.optim == "radam":
        optim = RAdam(model.parameters(), lr=config.lr)
    elif config.optim == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optim == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
        
    return optim

# Warmup and scheduler
def build_warmup_and_scheduler(config, optim):
    if config.warmup == "none":
        warmup = None
    elif config.warmup == "linear":
        warmup = pw.LinearWarmup(optim, warmup_period=config.warmup_period)
        
    if config.scheduler == "none":
        scheduler = None
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config.epoch, eta_min=config.lr_min)
    
    return warmup, scheduler

def build_loss_fn(config):
    if config.loss == "l2":
        loss_fn = nn.MSELoss()
    elif config.loss == "l1":
        loss_fn = nn.L1Loss()
    elif config.loss == "hybrid":
        loss_fn = HybridLoss(config)
    elif config.loss == "hybrid2":
        loss_fn = HybridLoss2(config)
    elif config.loss == "crossentropy":
        loss_fn = nn.CrossEntropyLoss()
    elif config.loss == "vqvae":
        loss_fn = VectorQuantizedVAELoss(config)
    elif config.loss == "vqvae2":
        loss_fn = VQVAELoss(config)
    elif config.loss == "fsq":
        loss_fn = FSQLoss(config)

    return loss_fn

