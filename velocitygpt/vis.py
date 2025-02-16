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
from .utils import _to_sequence2
from .datasets import *

def sample(model, context, length, config, cls=None, well_pos=None, well_token=None, dip=None, refl=None, dip_well=None, init=None):
    outputs = context.to(config.device) # add batch so shape [seq len, batch]
    pad = torch.zeros(config.n_concat_token, outputs.shape[-1], dtype=torch.long).to(config.device)  # to pad prev output
    with torch.no_grad():
        for l in range(length):
            if dip is not None:
                d = dip[:, :-(length-l-1)] if l < (length - 1) else dip
            else:
                d = None
            if refl is not None:
                if config.prepend_refl:
                    r = refl
                else:
                    r = refl[:-(length-l-1)] if l < (length - 1) else refl
            else:
                r = None
            if not config.classify:
                logits = model(torch.cat((outputs, pad), dim=0), cls, well_pos, well_token, d, r, dip_well, init)
            else:
                clf_logits, logits = model(torch.cat((outputs, pad), dim=0), cls, well_pos, well_token, d, r, dip_well, init)
            logits = logits[-config.n_concat_token:, :, :]# / 1
            probs = F.softmax(logits, dim=-1)
            pred = []
            for i in range(probs.shape[1]):
                for j in range(probs.shape[0]):
                    pred.append(torch.multinomial(probs[j, i], num_samples=1))
            pred = torch.tensor(pred).reshape(probs.shape[1], probs.shape[0]).transpose(0, 1)
            outputs = torch.cat((outputs, pred.to(config.device)), dim=0)
            
        if config.classify:
            pred_cls = clf_logits.argmax(-1)
        else:
            pred_cls = 0
            
    return outputs, pred_cls

def plot_example(vqvae_model, vqvae_refl_model, model, data, scaler1, pad, config, idx, idx_gen=[1], cls=None, 
                 well_pos=None, dip=None, scaler=1, log=False, prefix=0):
    device = config.device
    dx = 1
    dt = 0.001
    x0 = 0.
    
    idx_gen = idx_gen * len(idx)
    if config.cls_token and cls is None:
        cls = data.data['cls'][idx].to(config.device)
    elif not config.cls_token:
        cls = torch.zeros(len(idx)).to(config.device)
    
    model.eval()

    for i in range(len(idx)): 
        if config.dataset_type == "fld2":
            data.transform.transforms = [t for t in data.transform.transforms if 
                                         any([isinstance(t, Normalization), 
                                              isinstance(t, GaussianFilter)])]
            inputs = data[idx[i]]['input']['tensor'].unsqueeze(0).to(config.device)
            labels = inputs.clone()
            if config.vqvae_refl_dir is not None:
                refl = data[idx[i]]['label']['tensor'].unsqueeze(0).to(config.device)
            if config.use_init_prob:
                init = data[idx[i]]['label2']['tensor'].unsqueeze(0).to(config.device)
        else:
            inputs = data.data['input'][[idx[i]]].to(config.device)
            labels = data.data['label'][[idx[i]]].to(config.device)
            if config.vqvae_refl_dir is not None:
                refl = data.data['refl'][[idx[i]]].to(config.device)
        well_pos_inp = None
        well_token = None
        if config.well_cond_prob > 0:
            if well_pos is None:
                well_pos_inp = torch.tensor([config.image_size[0]]).to(config.device)
                well_token = config.vocab_size * torch.ones((config.latent_dim[1], 1)).long().to(config.device)
            elif well_pos is not None:
                print(well_pos[[i]])
                well_pos_inp = well_pos[[i]]
                well_vel = inputs.clone()
                well_vel = well_vel[range(len(well_pos_inp)), well_pos_inp]
                well_vel = well_vel.unsqueeze(1).repeat(1, config.image_size[0], 1)
                well_token = vqvae_model.encode(well_vel.unsqueeze(1))
                well_token = well_token[:, 0, :].transpose(0, 1)
                
        if dip is None:
            dip = len(config.dip_bins) * torch.ones(1, config.max_length).long().to(config.device)
            
        if config.add_dip_to_well and config.use_dip and config.well_cond_prob:
            well_pos2 = torch.clamp(well_pos_inp.clone(), max=config.image_size[0]-1) 
            dip_well = dip.clone()
            dip_well = dip_well.view(dip_well.shape[0], config.latent_dim[1], config.latent_dim[0])[range(len(well_pos2)), :, well_pos2//config.factor]
            dip_well = dip_well.view(dip_well.shape[0], -1)
        else:
            dip_well = None
        
        inputs = inputs[:, :, :idx_gen[i]]

        # Transform to discrete
        #         inputs, orig_shape = _to_sequence(inputs)
        with torch.no_grad():
            latents = vqvae_model.encode(inputs.unsqueeze(1))
            if config.vqvae_refl_dir is not None:
                latents_refl = vqvae_refl_model.encode(refl.unsqueeze(1))
                latents_refl, orig_shape = _to_sequence2(latents_refl)
            else:
                latents_refl = None
            if config.use_init_prob:
                latents_init = vqvae_model.encode(init.unsqueeze(1))
                latents_init, orig_shape = _to_sequence2(latents_init)
            else:
                latents_init = None
            
        # Get predictions
        latents, orig_shape = _to_sequence2(latents)
        length = config.max_length - latents.shape[0]
        preds, pred_cls = sample(model, latents, length//config.n_concat_token, config, cls[[i]], 
                                 well_pos_inp, well_token, dip, latents_refl, dip_well, latents_init)

        # # Transform back to image
        orig_shape = (orig_shape[0], orig_shape[1], int(orig_shape[2]+(length/config.latent_dim[0])))
        preds = _to_sequence2(preds, inv=True, orig_shape=orig_shape)
        with torch.no_grad():
            preds = vqvae_model.decode(preds).squeeze(1)
        # #         preds = _to_sequence(preds, inv=True, orig_shape=orig_shape)
        
        input_transformed = labels.clone() 
        if config.dataset_type == "fld2":
            input_transformed = data.denormalize({**data[idx[i]]['input'], 'tensor': input_transformed.cpu()})
            sample_output = data.denormalize({**data[idx[i]]['input'], 'tensor': preds.cpu()})
            labels = data.denormalize({**data[idx[i]]['input'], 'tensor': labels.cpu()})
        else:
            input_transformed = ((input_transformed / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]
            sample_output = ((preds  / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]
            labels = ((labels  / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]

        input_transformed[:, :, idx_gen[i]:] = 0
        
        if config.image_size != config.orig_image_size and config.revert:
            if config.pad_input:
                img_orig_shape = (1, config.orig_image_size[0], config.orig_image_size[1])
                input_transformed = pad_input(input_transformed, 
                                              pad, 
                                              inv=True, 
                                              orig_shape=img_orig_shape).cpu()
                sample_output = pad_input(sample_output, 
                                          pad, 
                                          inv=True, 
                                          orig_shape=img_orig_shape).cpu()
                labels = pad_input(labels, 
                                   pad, 
                                   inv=True, 
                                   orig_shape=img_orig_shape).cpu()
            else:
                sample_output = torch.tensor(resize_image(preds, 
                                                           inv=True, 
                                                           orig_image_size=config.orig_image_size))
                labels = torch.tensor(resize_image(labels,
                                                   inv=True, 
                                                   orig_image_size=config.orig_image_size))
                input_transformed = torch.tensor(resize_image(input_transformed,
                                                              inv=True, 
                                                              orig_image_size=config.orig_image_size))

        
        X = input_transformed.squeeze().cpu()
        y = sample_output.squeeze().cpu()
        z = labels.squeeze().cpu()
        mask = torch.zeros(1, X.shape[-1])
        mask[:, idx_gen[i]:] = 1
        
        f, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=False)
        im1 = ax[0].imshow(X.T * scaler, aspect=1, vmin=1500, vmax=4500, cmap='terrain')
        ax[0].set_title("Input")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Z")
        f.colorbar(im1, fraction=0.046, pad=0.04)
        
        im2 = ax[1].imshow(y.T * scaler, aspect=1, vmin=1500, vmax=4500, cmap='terrain')
        selected_y = y * (1 - mask.float().unsqueeze(1))
        selected_z = z * (1 - mask.float().unsqueeze(1))
        error = nn.MSELoss()(selected_y, selected_z)
        ax[1].set_title("Output (MSE: {:.3})".format(error))
        ax[1].set_xlabel("X")
        f.colorbar(im2, fraction=0.046, pad=0.04)
        
        im3 = ax[2].imshow(z.T * scaler, aspect=1, vmin=1500, vmax=4500, cmap='terrain')
        ax[2].set_title("Label")
        ax[2].set_xlabel("X")
        f.colorbar(im3, fraction=0.046, pad=0.04)
        
        diff = (z - y)
        im4 = ax[3].imshow(diff.T * scaler, aspect=1, vmin=-500, vmax=500, cmap='terrain')
        ax[3].set_title("(Label - Output)")
        ax[3].set_xlabel("X")
        f.colorbar(im4, fraction=0.046, pad=0.04)
        
        f.savefig(os.path.join(config.parent_dir, "example_prediction_{}_{}.png").format(prefix, i+1))
        if log:
            wandb.log({"plot_{}_{}".format(prefix, i+1): f})
            
        print("Predicted class: {}".format(pred_cls))
    
    # Revert back to original dropout
    set_dropout_prob(model, config.hidden_dropout_prob)

def plot_example2(model, data, scaler1, pad, config, idx, log=False, prefix=0):
    device = config.device
    dx = 1
    dt = 0.001
    x0 = 0.
        
    model.eval()
    
    training_stage = [config.training_stage, 'vqvae-training-refl']
    input_key = ['input', 'label']
    for i in range(len(idx)): 
        for j, ik, ts in zip(range(config.input_dim), input_key, training_stage):
            if config.dataset_type == "fld2":
                data.transform.transforms = [t for t in data.transform.transforms if 
                                            any([isinstance(t, Normalization)])]
                inputs = torch.zeros((1, config.input_dim, *config.image_size), device=config.device)
                inputs[:, j] = data[idx[i]][ik]['tensor'].to(config.device)
                labels = inputs.clone()
            else:
                inputs = data.data['input'][[idx[i]]].to(config.device)
                labels = data.data['label'][[idx[i]]].to(config.device)
                inputs = inputs.unsqueeze(1)
            
    #         inputs = _to_sequence(inputs, config)
            with torch.no_grad():
                if config.vq_type == "vqvae":
                    x_tilde, z_e_x, z_q_x = model(inputs)
                    latents = model.encode(inputs)
                elif config.vq_type == "vqvae2":
                    x_tilde, latent_loss = model(inputs)
                    _, _, _, *latents = model.encode(inputs)
            
    #         inputs = _to_sequence(inputs, config, inv=True)
            sample_output = x_tilde#.squeeze(1)
            if config.dataset_type == "fld2":
                inputs = data.denormalize({**data[idx[i]][ik], 'tensor': inputs.cpu()})
                sample_output = data.denormalize({**data[idx[i]][ik], 'tensor': sample_output.cpu()})
                labels = data.denormalize({**data[idx[i]][ik], 'tensor': labels.cpu()})
            else:
                inputs = ((inputs / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]
                sample_output = ((sample_output  / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]
                labels = ((labels  / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]
            if config.image_size != config.orig_image_size and config.revert:
                if config.pad_input:
                    orig_shape = (1, config.orig_image_size[0], config.orig_image_size[1])
                    inputs = pad_input(inputs, pad, inv=True, orig_shape=orig_shape)
                    sample_output = pad_input(sample_output, pad, inv=True, orig_shape=orig_shape)
                    labels = pad_input(labels, pad, inv=True, orig_shape=orig_shape)
                else:
                    inputs = torch.tensor(resize_image2(inputs[0].cpu().numpy(), 
                                                        config.orig_image_size))
                    sample_output = torch.tensor(resize_image2(sample_output[0].cpu().numpy(), 
                                                            config.orig_image_size))
                    labels = torch.tensor(resize_image2(labels[0].cpu().numpy(), 
                                                        config.orig_image_size))
            
            X = inputs[:, j].cpu()
            y = sample_output[:, j].cpu()
            z = labels[:, j].cpu()

            vlims = {"vqvae-training": [1500, 4500], "vqvae-training-refl": [-1, 1]}
            vlims_diff = {"vqvae-training": [-500, 500], "vqvae-training-refl": [-0.2, 0.2]}
            cmaps = {"vqvae-training": "terrain", "vqvae-training-refl": "Greys"}
            scalers_2 = {'syn1': 10, 'fld1': 10, 'fld2': 1e-2}
            scalers = {"vqvae-training": 1, "vqvae-training-refl": scalers_2[config.dataset_type]}
            
            f, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=False)
            im1 = ax[0].imshow(X.detach().T * scalers[ts], 
                            aspect=1, 
                            vmin=vlims[ts][0], 
                            vmax=vlims[ts][1], 
                            cmap=cmaps[ts])
            ax[0].set_title("Input")
            ax[0].set_xlabel("X")
            ax[0].set_ylabel("Z")
            f.colorbar(im1, fraction=0.046, pad=0.04)
            
            im2 = ax[1].imshow(y.detach().T * scalers[ts],
                            aspect=1, 
                            vmin=vlims[ts][0], 
                            vmax=vlims[ts][1], 
                            cmap=cmaps[ts])
            error = nn.MSELoss()(y, z)
            ax[1].set_title("Output (MSE: {:.3})".format(error))
            ax[1].set_xlabel("X")
            f.colorbar(im2, fraction=0.046, pad=0.04)
            
            im3 = ax[2].imshow(z.detach().T * scalers[ts],
                            aspect=1, 
                            vmin=vlims[ts][0], 
                            vmax=vlims[ts][1], 
                            cmap=cmaps[ts])
            ax[2].set_title("Label")
            ax[2].set_xlabel("X")
            f.colorbar(im3, fraction=0.046, pad=0.04)
            
            diff = (z - y)
            im4 = ax[3].imshow(diff.detach().T * scalers[ts], 
                            aspect=1, 
                            vmin=vlims_diff[ts][0], 
                            vmax=vlims_diff[ts][1], 
                            cmap=cmaps[ts])
            ax[3].set_title("(Label - Output)")
            ax[3].set_xlabel("X")
            f.colorbar(im4, fraction=0.046, pad=0.04)

            f.savefig(os.path.join(config.parent_dir, "example_prediction_{}_{}_{}.png").format(prefix, i+1, j+1))
            if log:
                wandb.log({"plot_{}_{}_{}".format(prefix, i+1, j+1): f})
                
            print(latents)

def plot_example3(model, data, scaler1, pad, config, idx, scaler=1, log=False, prefix=0):
    device = config.device
    dx = 1
    dt = 0.001
    x0 = 0.
        
    model.eval()

    for i in range(len(idx)): 
        inputs = data.data['input'][[idx[i]]].to(config.device)
        labels = data.data['label'][[idx[i]]].to(config.device)
        
#         inputs = _to_sequence(inputs, config)
        with torch.no_grad():
            x_tilde = model(inputs.unsqueeze(1))
        
#         inputs = _to_sequence(inputs, config, inv=True)
        sample_output = x_tilde.squeeze(1)
        inputs = ((inputs / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]
        sample_output = ((sample_output  / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]
        labels = ((labels  / config.scaler2) + config.scaler3) * scaler1[[idx[i]]][:, None, None]
        if config.image_size != config.orig_image_size and config.revert:
            if config.pad_input:
                orig_shape = (1, config.orig_image_size[0], config.orig_image_size[1])
                inputs = pad_input(inputs, pad, inv=True, orig_shape=orig_shape)
                sample_output = pad_input(sample_output, pad, inv=True, orig_shape=orig_shape)
                labels = pad_input(labels, pad, inv=True, orig_shape=orig_shape)
            else:
                inputs = torch.tensor(resize_image2(inputs[0].cpu().numpy(), 
                                                    config.orig_image_size))
                sample_output = torch.tensor(resize_image2(sample_output[0].cpu().numpy(), 
                                                           config.orig_image_size))
                labels = torch.tensor(resize_image2(labels[0].cpu().numpy(), 
                                                    config.orig_image_size))
        
        X = inputs.squeeze().cpu()
        y = sample_output.squeeze().cpu()
        z = labels.squeeze().cpu()
                
        f, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=False, sharex=False)
        im1 = ax[0].imshow(X.detach().T * scaler, aspect=1, vmin=1500, vmax=4500, cmap='terrain')
        ax[0].set_title("Input")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Z")
        f.colorbar(im1, fraction=0.046, pad=0.04)
        
        im2 = ax[1].imshow(y.detach().T * scaler, aspect=1, vmin=1500, vmax=4500, cmap='terrain')
        error = nn.MSELoss()(y, z)
        ax[1].set_title("Output (MSE: {:.3})".format(error))
        ax[1].set_xlabel("X")
        f.colorbar(im2, fraction=0.046, pad=0.04)
        
        im3 = ax[2].imshow(z.detach().T * scaler, aspect=1, vmin=1500, vmax=4500, cmap='terrain')
        ax[2].set_title("Label")
        ax[2].set_xlabel("X")
        f.colorbar(im3, fraction=0.046, pad=0.04)
        
        diff = (z - y)
        im4 = ax[3].imshow(diff.detach().T * scaler, aspect=1, vmin=-500, vmax=500, cmap='terrain')
        ax[3].set_title("(Label - Output)")
        ax[3].set_xlabel("X")
        f.colorbar(im4, fraction=0.046, pad=0.04)
        
        f.savefig(os.path.join(config.parent_dir, "example_prediction_{}_{}.png").format(prefix, i+1))
        if log:
            wandb.log({"plot_{}_{}".format(prefix, i+1): f})