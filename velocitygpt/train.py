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

from .pytorchtools import EarlyStopping
from .utils import *
from .utils import _to_sequence2
from .quantizer import FSQ

def run_velenc(model, optim, warmup, scheduler, loss_fn, train_dataloader, test_dataloader, scaler1, config, 
               plot=False, f=None, ax=None, verbose=True):
    epochs = config.epoch
    device = config.device
    total_time = time.time()
    avg_train_loss = []
    avg_valid_loss = []
    avg_train_psnr = []
    avg_valid_psnr = []
    avg_train_ssim = []
    avg_valid_ssim = []
    avg_train_cu = []
    avg_valid_cu = []
    time_per_epoch = []
    lr_epoch = []
    if config.patience is not None:
        checkpoint = os.path.join(config.parent_dir, str(os.getpid())+"checkpoint.pt")
        early_stopping = EarlyStopping(patience=config.patience, verbose=False, path=checkpoint)
    
    try:
        start_epoch = get_previous_epoch_count(config) if hasattr(config, 'cont_dir') and config.cont_dir else 0
        loop_epoch = tqdm(range(start_epoch, epochs))
        for epoch in loop_epoch:
            epoch_time = time.time()
            lr_epoch.append(get_lr(optim))
            model.train()
            # setup loop with TQDM and dataloader
            if verbose:
                loop_train = tqdm(train_dataloader, leave=True, position=0)
            else:
                loop_train = train_dataloader
            losses_train = 0
            psnr_train = 0
            ssim_train = 0
            cu_train = 0
            for i, batch in enumerate(loop_train):
                # initialize calculated gradients (from prev step)
                optim.zero_grad()

                # pull all tensor batches required for training
                if config.dataset_type in ["fld2", "syn2"]:
                    batch['input'] = {k: v.to(device) for k, v in batch['input'].items()}
                    batch['label'] = {k: v.to(device) for k, v in batch['label'].items()}
                    inputs = batch['input']['tensor']
                    labels = batch['label']['tensor']
                    if config.input_dim == 1:
                        inputs = inputs.unsqueeze(1)
                        labels = inputs.clone()
                    elif config.input_dim == 2:
                        inputs = torch.stack((inputs, labels), dim=1)
                        if config.input_rand_mask:
                            # Create a balanced distribution of mask patterns
                            # Each sample will get one of three patterns with equal probability:
                            # [1,0], [0,1], or [1,1]
                            pattern_choice = torch.randint(0, 3, (inputs.shape[0],))
                            rand_mask = torch.zeros(inputs.shape[0], config.input_dim)
                            
                            # Pattern [1,0]
                            rand_mask[pattern_choice == 0, 0] = 1.0
                            
                            # Pattern [0,1]
                            rand_mask[pattern_choice == 1, 1] = 1.0
                            
                            # Pattern [1,1]
                            rand_mask[pattern_choice == 2, 0] = 1.0
                            rand_mask[pattern_choice == 2, 1] = 1.0
                        else:
                            rand_mask = torch.ones(inputs.shape[0], config.input_dim)
                        rand_mask = rand_mask.reshape(*rand_mask.shape, 1, 1)
                        inputs = inputs * rand_mask.to(inputs.device)
                        labels = inputs.clone()
                else:
                    inputs = batch['input'].to(config.device)
                    labels = batch['label'].to(config.device)

                # process
    #             inputs = _to_sequence(inputs, config)
                if config.vq_type == "vqvae":
                    x_tilde, z_e_x, aux_loss, latents = model(inputs, return_latents=True)
                    x_tilde = x_tilde * rand_mask.to(inputs.device) if config.input_dim == 2 else x_tilde
                    loss = loss_fn(x_tilde, inputs, aux_loss)
                elif config.vq_type == "vqvae2":
                    x_tilde, latent_loss = model(inputs.unsqueeze(1))
                    loss = loss_fn(x_tilde, inputs.unsqueeze(1), latent_loss)

                loss.backward()

                # update parameters
                optim.step()
                
                # calculate metrics
                losses_train += loss.item()
                with torch.no_grad():
    #                 outputs = _to_sequence(x_tilde, config, inv=True)
                    if config.input_dim == 1:
                        outputs = x_tilde.squeeze(1)
                        labels = labels.squeeze(1)
                    elif config.input_dim == 2:
                        outputs, outputs2 = x_tilde[:, 0], x_tilde[:, 1]
                        labels, labels2 = labels[:, 0], labels[:, 1]
                    if config.dataset_type in ["fld2", "syn2"]:
                        selected_outputs = train_dataloader.dataset.denormalize({**batch['input'], 'tensor': outputs})
                        selected_labels = train_dataloader.dataset.denormalize({**batch['input'], 'tensor': labels})
                        if config.input_dim == 1:
                            selected_outputs = selected_outputs.unsqueeze(1)
                            selected_labels = selected_labels.unsqueeze(1)
                        elif config.input_dim == 2:
                            selected_outputs2 = train_dataloader.dataset.denormalize({**batch['label'], 'tensor': outputs2})
                            selected_labels2 = train_dataloader.dataset.denormalize({**batch['label'], 'tensor': labels2})   
                            selected_outputs = torch.stack((selected_outputs, selected_outputs2), dim=1)
                            selected_labels = torch.stack((selected_labels, selected_labels2), dim=1)
                    else:     
                        selected_outputs = (outputs.unsqueeze(1) / config.scaler2) + config.scaler3
                        selected_labels = (labels.unsqueeze(1) / config.scaler2) + config.scaler3
                    idx_start = i*config.batch_size
                    idx_end = (i+1)*config.batch_size
                    psnr_train += PSNR((selected_outputs * scaler1[0][idx_start:idx_end][:, None, None, None]).ravel(),
                                    (selected_labels * scaler1[0][idx_start:idx_end][:, None, None, None]).ravel())
                    ssim_train += ssim((selected_outputs * scaler1[0][idx_start:idx_end][:, None, None, None]) + 1, 
                                    (selected_labels * scaler1[0][idx_start:idx_end][:, None, None, None]) + 1, 
                                    data_range=2, size_average=True)
                    cu_train += len(latents.unique()) / config.K
                if verbose:
                    loop_train.set_description(f'Epoch {epoch}')
                    loop_train.set_postfix(loss=loss.item())
                
                if i == 0:
                    nvsmi = nvidia_smi.getInstance()
                    gpu_memory_used = nvsmi.DeviceQuery('memory.used')['gpu'][0]['fb_memory_usage']['used']
                            
            model.eval()
            if verbose:
                loop_valid = tqdm(test_dataloader, leave=True, position=0)
            else:
                loop_valid = test_dataloader
            losses_valid = 0
            psnr_valid = 0
            ssim_valid = 0
            cu_valid = 0
            with torch.no_grad():
                for i, batch in enumerate(loop_valid):
                    # pull all tensor batches required for training
                    if config.dataset_type in ["fld2", "syn2"]:
                        batch['input'] = {k: v.to(device) for k, v in batch['input'].items()}
                        batch['label'] = {k: v.to(device) for k, v in batch['label'].items()}
                        inputs = batch['input']['tensor']
                        labels = batch['label']['tensor']
                        if config.input_dim == 1:
                            inputs = inputs.unsqueeze(1)
                            labels = inputs.clone()
                        elif config.input_dim == 2:
                            inputs = torch.stack((inputs, labels), dim=1)
                            if config.input_rand_mask:
                                # Create a balanced distribution of mask patterns
                                # Each sample will get one of three patterns with equal probability:
                                # [1,0], [0,1], or [1,1]
                                pattern_choice = torch.randint(0, 3, (inputs.shape[0],))
                                rand_mask = torch.zeros(inputs.shape[0], config.input_dim)
                                
                                # Pattern [1,0]
                                rand_mask[pattern_choice == 0, 0] = 1.0
                                
                                # Pattern [0,1]
                                rand_mask[pattern_choice == 1, 1] = 1.0
                                
                                # Pattern [1,1]
                                rand_mask[pattern_choice == 2, 0] = 1.0
                                rand_mask[pattern_choice == 2, 1] = 1.0
                            else:
                                rand_mask = torch.ones(inputs.shape[0], config.input_dim)
                            rand_mask = rand_mask.reshape(*rand_mask.shape, 1, 1)
                            inputs = inputs * rand_mask.to(inputs.device)
                            labels = inputs.clone()
                    else:
                        inputs = batch['input'].to(config.device)
                        labels = batch['label'].to(config.device)

                    # process
    #                 inputs = _to_sequence(inputs, config)
                    if config.vq_type == "vqvae":
                        x_tilde, z_e_x, aux_loss, latents = model(inputs, return_latents=True)
                        x_tilde = x_tilde * rand_mask.to(inputs.device) if config.input_dim == 2 else x_tilde
                        loss = loss_fn(x_tilde, inputs, aux_loss)
                    elif config.vq_type == "vqvae2":
                        x_tilde, latent_loss = model(inputs)
                        loss = loss_fn(x_tilde, inputs, latent_loss)
                    
                    # calculate metrics
                    losses_valid += loss.item()
    #                 outputs = _to_sequence(x_tilde, config, inv=True)
                    if config.input_dim == 1:
                        outputs = x_tilde.squeeze(1)
                        labels = labels.squeeze(1)
                    elif config.input_dim == 2:
                        outputs, outputs2 = x_tilde[:, 0], x_tilde[:, 1]
                        labels, labels2 = labels[:, 0], labels[:, 1]
                    if config.dataset_type in ["fld2", "syn2"]:
                        selected_outputs = test_dataloader.dataset.denormalize({**batch['input'], 'tensor': outputs})
                        selected_labels = test_dataloader.dataset.denormalize({**batch['input'], 'tensor': labels})
                        if config.input_dim == 1:
                            selected_outputs = selected_outputs.unsqueeze(1)
                            selected_labels = selected_labels.unsqueeze(1)
                        elif config.input_dim == 2:
                            selected_outputs2 = test_dataloader.dataset.denormalize({**batch['label'], 'tensor': outputs2})
                            selected_labels2 = test_dataloader.dataset.denormalize({**batch['label'], 'tensor': labels2})   
                            selected_outputs = torch.stack((selected_outputs, selected_outputs2), dim=1)
                            selected_labels = torch.stack((selected_labels, selected_labels2), dim=1)
                    else:
                        selected_outputs = (outputs.unsqueeze(1) / config.scaler2) + config.scaler3
                        selected_labels = (labels.unsqueeze(1) / config.scaler2) + config.scaler3
                    idx_start = i*config.batch_size
                    idx_end = (i+1)*config.batch_size
                    psnr_valid += PSNR((selected_outputs * scaler1[1][idx_start:idx_end][:, None, None, None]).ravel(),
                                    (selected_labels * scaler1[1][idx_start:idx_end][:, None, None, None]).ravel())
                    ssim_valid += ssim((selected_outputs * scaler1[1][idx_start:idx_end][:, None, None, None]) + 1, 
                                    (selected_labels * scaler1[1][idx_start:idx_end][:, None, None, None]) + 1, 
                                    data_range=2, size_average=True)
                    if verbose:
                        loop_valid.set_description(f'Validation {epoch}')
                        loop_valid.set_postfix(loss=loss.item())
                    cu_valid += len(latents.unique()) / config.K
                    

            avg_train_loss.append(losses_train / len(train_dataloader))
            avg_valid_loss.append(losses_valid / len(test_dataloader))
            avg_train_psnr.append(psnr_train / len(train_dataloader))
            avg_valid_psnr.append(psnr_valid / len(test_dataloader))
            avg_train_ssim.append(ssim_train / len(train_dataloader))
            avg_valid_ssim.append(ssim_valid / len(test_dataloader))
            time_per_epoch.append(time.time() - epoch_time)
            avg_train_cu.append(cu_train / len(train_dataloader))
            avg_valid_cu.append(cu_valid / len(test_dataloader))

            # warmup and scheduler
            if scheduler is not None:
                if warmup is not None:
                    with warmup.dampening():
                        if config.scheduler == "reduce_on_plateau":
                            scheduler.step(avg_valid_loss[-1])
                        else:
                            scheduler.step()
                else:
                    if config.scheduler == "reduce_on_plateau":
                        scheduler.step(avg_valid_loss[-1])
                    else:
                        scheduler.step()
            elif warmup is not None:
                with warmup.dampening():
                    pass
            
            loop_epoch.set_description(f'Epoch {epoch}')
            loop_epoch.set_postfix(avg_valid_loss=avg_valid_loss[-1])
            
            if verbose:
                print("Epoch time: {:.2f} s".format(time.time() - epoch_time))
                print("Total time elapsed: {:.2f} s".format(time.time() - total_time))
                print("---------------------------------------")
            
            if config.wandb_log:
                wandb.log({"avg_train_loss": avg_train_loss[-1], 
                        "avg_valid_loss": avg_valid_loss[-1], 
                        "avg_train_psnr": avg_train_psnr[-1], 
                        "avg_valid_psnr": avg_valid_psnr[-1], 
                        "avg_train_ssim": avg_train_ssim[-1], 
                        "avg_valid_ssim": avg_valid_ssim[-1], 
                        "time_per_epoch": time_per_epoch[-1],
                        "gpu_memory_used": gpu_memory_used, 
                        "epoch": epoch, 
                        "lr_epoch": lr_epoch[-1],
                        "avg_train_cu": avg_train_cu[-1], 
                        "avg_valid_cu": avg_valid_cu[-1]})
            
            if plot:
                ax.cla()
                ax.plot(np.arange(1, epoch+2), avg_train_loss,'b', label='Training Loss')
                ax.plot(np.arange(1, epoch+2), avg_valid_loss, 'orange', label='Validation Loss')
                ax.legend()
                ax.set_title("Loss Curve")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Avg Loss")
                f.canvas.draw()

            if config.patience is not None:
                early_stopping(-avg_valid_ssim[-1], model, optim=optim, scheduler=scheduler)

                if early_stopping.early_stop:
                    print("Early stopping")
                    if config.wandb_log:
                        wandb.log({"avg_train_loss": avg_train_loss[epoch-config.patience], 
                                "avg_valid_loss": avg_valid_loss[epoch-config.patience], 
                                "avg_train_psnr": avg_train_psnr[epoch-config.patience], 
                                "avg_valid_psnr": avg_valid_psnr[epoch-config.patience], 
                                "avg_train_ssim": avg_train_ssim[epoch-config.patience], 
                                "avg_valid_ssim": avg_valid_ssim[epoch-config.patience], 
                                "avg_train_cu": avg_train_cu[epoch-config.patience], 
                                "avg_valid_cu": avg_valid_cu[epoch-config.patience]})
                    break
        
        if config.patience is not None:
            model.load_state_dict(torch.load(checkpoint)['model'])
            optim.load_state_dict(torch.load(checkpoint)['optim'])
            if scheduler is not None:
                scheduler.load_state_dict(torch.load(checkpoint)['scheduler'])
    
    except KeyboardInterrupt:
        print("Stopped training, returning to last checkpoint...")
        model.load_state_dict(torch.load(checkpoint)['model'])
        optim.load_state_dict(torch.load(checkpoint)['optim'])
        if scheduler is not None:
            scheduler.load_state_dict(torch.load(checkpoint)['scheduler'])
        
    return model, avg_train_loss, avg_valid_loss, time_per_epoch

def run_velgen(model, vqvae_model, vqvae_refl_model, optim, warmup, scheduler, loss_fn, train_dataloader, 
               test_dataloader, scaler1, config, plot=False, f=None, ax=None, verbose=True):
    epochs = config.epoch
    device = config.device
    total_time = time.time()
    avg_train_loss = []
    avg_valid_loss = []
    avg_train_psnr = []
    avg_valid_psnr = []
    avg_train_ssim = []
    avg_valid_ssim = []
    time_per_epoch = []
    lr_epoch = []
    avg_train_clf_loss = []
    avg_valid_clf_loss = []
    avg_train_gen_loss = []
    avg_valid_gen_loss = []
    avg_train_clf_acc = []
    avg_valid_clf_acc = []
    if config.patience is not None:
        checkpoint = os.path.join(config.parent_dir, str(os.getpid())+"checkpoint.pt")
        early_stopping = EarlyStopping(patience=config.patience, verbose=False, path=checkpoint)
    
    try:
        start_epoch = get_previous_epoch_count(config) if hasattr(config, 'cont_dir') and config.cont_dir else 0
        loop_epoch = tqdm(range(start_epoch, epochs))
        for epoch in loop_epoch:
            epoch_time = time.time()
            lr_epoch.append(get_lr(optim))
            model.train()
    #         teacher_forcing_ratio = calc_teacher_forcing_ratio(epoch, config)
    #         print("Teacher forcing ratio: {:.2f}".format(teacher_forcing_ratio))
            # setup loop with TQDM and dataloader
            if verbose:
                loop_train = tqdm(train_dataloader, leave=True, position=0)
            else:
                loop_train = train_dataloader
            losses_train = 0
            psnr_train = 0
            ssim_train = 0
            losses_clf_train = 0
            losses_gen_train = 0
            acc_clf_train = 0
            for i, batch in enumerate(loop_train):
                # initialize calculated gradients (from prev step)
                if i % config.accum_grad == 0 or i == 0:
                    optim.zero_grad()

                # pull all tensor batches required for training
                if config.dataset_type in ["fld2", "syn2"]:
                    batch['input'] = {k: v.to(device) for k, v in batch['input'].items()}
                    inputs = batch['input']['tensor']
                    labels = inputs.clone()
                    if config.vqvae_refl_dir is not None:
                        refl = batch['label']['tensor'].to(device)
                    cls = None
                    if config.use_dip:
                        dips = batch['dip_seq'].to(config.device)
                        dips = dips.to(config.device)
                    else:
                        dips = None
                    if config.use_init_prob:
                        init = batch['label2']['tensor'].to(device)
                    else:
                        init = None
                else:
                    inputs = batch['input'].to(config.device)
                    labels = batch['label'].to(config.device)
                    if config.vqvae_refl_dir is not None:
                        refl = batch['refl'].to(config.device)
                    if config.classify or config.cls_token:
                        cls = batch['cls'].to(config.device)
                    else:
                        cls = None
                    if config.use_dip:
                        dips = batch['dip_seq'].to(config.device)
                        dips = dips.to(config.device)
                    else:
                        dips = None

                # process  
                with torch.no_grad():
                    if config.vqvae_dir == config.vqvae_refl_dir:
                        latents = vqvae_model.encode(torch.stack((inputs, torch.zeros_like(inputs)), dim=1))
                    else:
                        latents = vqvae_model.encode(inputs.unsqueeze(1))
                    if config.well_cond_prob > 0:
                        with_well_mask = (torch.rand(len(inputs)) < config.well_cond_prob)
                        well_pos = torch.randint(high=config.image_size[0], size=(len(inputs),)).to(config.device)
                        
                        # Extract 1d velocity
                        well_vel = inputs.clone()
                        well_vel = well_vel[range(len(well_pos)), well_pos]
                        well_vel = well_vel.unsqueeze(1).repeat(1, config.image_size[0], 1)
                        latents_well = vqvae_model.encode(well_vel.unsqueeze(1))
                        latents_well = latents_well[:, config.latent_dim[0]//2, :]
                        
                        # Change unselected well_pos and latents_well to OOV
                        well_pos[~with_well_mask] = config.image_size[0]
                        latents_well[~with_well_mask] = config.vocab_size
                        
                        latents_well = latents_well.transpose(0, 1)
                    else:
                        well_pos, latents_well = None, None
                        
                    if config.use_dip:
                        with_dip_mask = (torch.rand(len(inputs)) < config.use_dip_prob)
                        dips[~with_dip_mask] = len(config.dip_bins)
                        
                    if config.add_dip_to_well and config.use_dip and config.well_cond_prob:
                        well_pos2 = torch.clamp(well_pos.clone(), max=config.image_size[0]-1) 
                        dip_well = dips.clone()
                        dip_well = dip_well.view(dip_well.shape[0], config.latent_dim[0], config.latent_dim[1])[range(len(well_pos2)), :, well_pos2//config.factor]
                        dip_well = dip_well.view(dip_well.shape[0], -1)
                    else:
                        dip_well = None
                        
                    if config.vqvae_refl_dir is not None:
                        if config.vqvae_dir == config.vqvae_refl_dir:
                            latents_refl = vqvae_refl_model.encode(torch.stack((torch.zeros_like(refl), refl), dim=1))
                        else:
                            latents_refl = vqvae_refl_model.encode(refl.unsqueeze(1))
                        latents_refl, orig_shape = _to_sequence2(latents_refl)
                        with_refl_mask = (torch.rand(len(inputs)) < config.use_refl_prob)
                        latents_refl[:, ~with_refl_mask] = config.refl_vocab_size
                    else:
                        latents_refl = None

                    if config.use_init_prob:
                        if config.vqvae_dir == config.vqvae_refl_dir:
                            latents_init = vqvae_model.encode(torch.stack((init, torch.zeros_like(init)), dim=1))
                        else:
                            latents_init = vqvae_model.encode(init.unsqueeze(1))
                        latents_init, _ = _to_sequence2(latents_init)
                        with_init_mask = (torch.rand(len(inputs)) < config.use_init_prob)
                        latents_init[:, ~with_init_mask] = config.vocab_size
                    else:
                        latents_init = None
                    
                if config.flip_train:
                    latents2, orig_shape = _to_sequence2(torch.flip(latents, dims=(1,)))
                latents, orig_shape = _to_sequence2(latents)
                
                if not config.classify:
                    outputs = model(latents, cls, well_pos, latents_well, dips, latents_refl, dip_well, latents_init)
                    loss = loss_fn(outputs.view(-1, outputs.size(-1)), 
                                latents.reshape(-1).long())
                    if config.flip_train:
                        outputs2 = model(latents2, cls, well_pos, latents_well, dips, latents_refl, dip_well, latents_init)
                        if config.flip_train_inv:
                            outputs2 = outputs2.reshape(orig_shape[1], orig_shape[2], *outputs2.shape[1:])
                            outputs2 = torch.flip(outputs2, dims=(1,)).reshape(-1, *outputs2.shape[1:])
                            loss2 = loss_fn(outputs2.view(-1, outputs2.size(-1)), 
                                            latents.reshape(-1))
                        else:
                            loss2 = loss_fn(outputs2.view(-1, outputs2.size(-1)), 
                                            latents2.reshape(-1))
                        
                        loss = loss + loss2
                            
                    loss_gen = torch.tensor([0])
                    loss_clf = torch.tensor([0])
                else:
                    clf_logits, outputs = model(latents, cls, well_pos, latents_well, dips, latents_refl, dip_well, latents_init)
                    
                    loss_gen = loss_fn(outputs.view(-1, outputs.size(-1)), 
                                    latents.reshape(-1))
                    loss_clf = loss_fn(clf_logits, cls)
                    loss = loss_gen + loss_clf

                loss.backward()

                # update parameters
                if (i+1) % config.accum_grad == 0:
                    optim.step()
                
                # calculate metrics
                losses_train += loss.item()
                losses_gen_train += loss_gen.item()
                losses_clf_train += loss_clf.item()
                with torch.no_grad():
                    outputs = _to_sequence2(outputs.argmax(-1), inv=True, orig_shape=orig_shape)
                    outputs = vqvae_model.decode(outputs).squeeze(1)
    #                 outputs = _to_sequence(outputs, inv=True, orig_shape=orig_shape)
                    if config.dataset_type in ["fld2", "syn2"]:
                        selected_outputs = train_dataloader.dataset.denormalize({**batch['input'], 'tensor': outputs})
                        selected_labels = train_dataloader.dataset.denormalize({**batch['input'], 'tensor': labels})
                        selected_outputs = selected_outputs.unsqueeze(1)
                        selected_labels = selected_labels.unsqueeze(1)                                
                    else:
                        selected_outputs = (outputs.unsqueeze(1) / config.scaler2) + config.scaler3
                        selected_labels = (labels.unsqueeze(1) / config.scaler2) + config.scaler3
                    idx_start = i*config.batch_size
                    idx_end = (i+1)*config.batch_size
                    psnr_train += PSNR((selected_outputs * scaler1[0][idx_start:idx_end][:, None, None, None]).ravel(),
                                    (selected_labels * scaler1[0][idx_start:idx_end][:, None, None, None]).ravel())
                    ssim_train += ssim((selected_outputs * scaler1[0][idx_start:idx_end][:, None, None, None]), 
                                    (selected_labels * scaler1[0][idx_start:idx_end][:, None, None, None]), 
                                    data_range=4500, size_average=True)
                    if config.classify:
                        clf_logits = clf_logits.argmax(-1)
                        acc_clf_train += (clf_logits == cls).sum().item() / len(cls)
                if verbose:
                    loop_train.set_description(f'Epoch {epoch}')
                    loop_train.set_postfix(loss=loss.item())
                
                if i == 0:
                    nvsmi = nvidia_smi.getInstance()
                    gpu_memory_used = nvsmi.DeviceQuery('memory.used')['gpu'][0]['fb_memory_usage']['used']
                            
            model.eval()
            if verbose:
                loop_valid = tqdm(test_dataloader, leave=True, position=0)
            else:
                loop_valid = test_dataloader
            losses_valid = 0
            psnr_valid = 0
            ssim_valid = 0
            losses_clf_valid = 0
            losses_gen_valid = 0
            acc_clf_valid = 0
            with torch.no_grad():
                for i, batch in enumerate(loop_valid):
                    # pull all tensor batches required for training
                    if config.dataset_type in ["fld2", "syn2"]:
                        batch['input'] = {k: v.to(device) for k, v in batch['input'].items()}
                        inputs = batch['input']['tensor']
                        labels = inputs.clone()
                        if config.vqvae_refl_dir is not None:
                            refl = batch['label']['tensor'].to(device)
                        cls = None
                        if config.use_dip:
                            dips = batch['dip_seq'].to(config.device)
                            dips = dips.to(config.device)
                        else:
                            dips = None
                        if config.use_init_prob:
                            init = batch['label2']['tensor'].to(device)
                        else:
                            init = None
                    else:
                        inputs = batch['input'].to(config.device)
                        labels = batch['label'].to(config.device) 
                        if config.vqvae_refl_dir is not None:
                            refl = batch['refl'].to(config.device)
                        if config.classify or config.cls_token:
                            cls = batch['cls'].to(config.device)
                        else:
                            cls = None
                        if config.use_dip:
                            dips = batch['dip_seq'].to(config.device)
                            dips = dips.to(config.device)
                        else:
                            dips = None

                    # process  
                    if config.vqvae_dir == config.vqvae_refl_dir:
                        latents = vqvae_model.encode(torch.stack((inputs, torch.zeros_like(inputs)), dim=1))
                    else:
                        latents = vqvae_model.encode(inputs.unsqueeze(1))
                    if config.well_cond_prob > 0:
                        with_well_mask = (torch.rand(len(inputs)) < config.well_cond_prob)
                        well_pos = torch.randint(high=config.image_size[0], size=(len(inputs),)).to(config.device)
                        
                        # Extract 1d velocity
                        well_vel = inputs.clone()
                        well_vel = well_vel[range(len(well_pos)), well_pos]
                        well_vel = well_vel.unsqueeze(1).repeat(1, config.image_size[0], 1)
                        latents_well = vqvae_model.encode(well_vel.unsqueeze(1))
                        latents_well = latents_well[:, config.latent_dim[0]//2, :]
                        
                        # Change unselected well_pos and latents_well to OOV
                        well_pos[~with_well_mask] = config.image_size[0]
                        latents_well[~with_well_mask] = config.vocab_size
                        
                        latents_well = latents_well.transpose(0, 1)
                    else:
                        well_pos, latents_well = None, None
                        
                    if config.use_dip:
                        with_dip_mask = (torch.rand(len(inputs)) < config.use_dip_prob)
                        dips[~with_dip_mask] = len(config.dip_bins)
                        
                    if config.add_dip_to_well and config.use_dip and config.well_cond_prob:
                        well_pos2 = torch.clamp(well_pos.clone(), max=config.image_size[0]-1) 
                        dip_well = dips.clone()
                        dip_well = dip_well.view(dip_well.shape[0], config.latent_dim[0], config.latent_dim[1])[range(len(well_pos2)), :, well_pos2//config.factor]
                        dip_well = dip_well.view(dip_well.shape[0], -1)
                    else:
                        dip_well = None
                        
                    if config.vqvae_refl_dir is not None:
                        if config.vqvae_dir == config.vqvae_refl_dir:
                            latents_refl = vqvae_refl_model.encode(torch.stack((torch.zeros_like(refl), refl), dim=1))
                        else:
                            latents_refl = vqvae_refl_model.encode(refl.unsqueeze(1))
                        latents_refl, orig_shape = _to_sequence2(latents_refl)
                        with_refl_mask = (torch.rand(len(inputs)) < config.use_refl_prob)
                        latents_refl[:, ~with_refl_mask] = config.refl_vocab_size
                    else:
                        latents_refl = None

                    if config.use_init_prob:
                        if config.vqvae_dir == config.vqvae_refl_dir:
                            latents_init = vqvae_model.encode(torch.stack((init, torch.zeros_like(init)), dim=1))
                        else:
                            latents_init = vqvae_model.encode(init.unsqueeze(1))
                        latents_init, _ = _to_sequence2(latents_init)
                        with_init_mask = (torch.rand(len(inputs)) < config.use_init_prob)
                        latents_init[:, ~with_init_mask] = config.vocab_size
                    else:
                        latents_init = None

                    if config.flip_train:
                        latents2, orig_shape = _to_sequence2(torch.flip(latents, dims=(1,)))
                    latents, orig_shape = _to_sequence2(latents)

                    if not config.classify:
                        outputs = model(latents, cls, well_pos, latents_well, dips, latents_refl, dip_well, latents_init)
                        loss = loss_fn(outputs.view(-1, outputs.size(-1)), 
                                    latents.reshape(-1).long())
                        if config.flip_train:
                            outputs2 = model(latents2, cls, well_pos, latents_well, dips, latents_refl, dip_well, latents_init)
                            if config.flip_train_inv:
                                outputs2 = outputs2.reshape(orig_shape[1], orig_shape[2], *outputs2.shape[1:])
                                outputs2 = torch.flip(outputs2, dims=(1,)).reshape(-1, *outputs2.shape[1:])
                                loss2 = loss_fn(outputs2.view(-1, outputs2.size(-1)), 
                                                latents.reshape(-1))
                            else:
                                loss2 = loss_fn(outputs2.view(-1, outputs2.size(-1)), 
                                                latents2.reshape(-1))

                            loss = loss + loss2
                            
                        loss_gen = torch.tensor([0])
                        loss_clf = torch.tensor([0])
                    else:
                        clf_logits, outputs = model(latents, cls, well_pos, latents_well, dips, latents_refl, dip_well, latents_init)

                        loss_gen = loss_fn(outputs.view(-1, outputs.size(-1)), 
                                        latents.reshape(-1))
                        loss_clf = loss_fn(clf_logits, cls)
                        loss = loss_gen + loss_clf
                    
                    # calculate metrics
                    losses_valid += loss.item()
                    losses_gen_valid += loss_gen.item()
                    losses_clf_valid += loss_clf.item()
                    outputs = _to_sequence2(outputs.argmax(-1), inv=True, orig_shape=orig_shape)
                    outputs = vqvae_model.decode(outputs).squeeze(1)
    #                 outputs = _to_sequence(outputs, inv=True, orig_shape=orig_shape)
                    if config.dataset_type in ["fld2", "syn2"]:
                        selected_outputs = test_dataloader.dataset.denormalize({**batch['input'], 'tensor': outputs})
                        selected_labels = test_dataloader.dataset.denormalize({**batch['input'], 'tensor': labels})
                        selected_outputs = selected_outputs.unsqueeze(1)
                        selected_labels = selected_labels.unsqueeze(1)                                
                    else:
                        selected_outputs = (outputs.unsqueeze(1) / config.scaler2) + config.scaler3
                        selected_labels = (labels.unsqueeze(1) / config.scaler2) + config.scaler3
                    idx_start = i*config.batch_size
                    idx_end = (i+1)*config.batch_size
                    psnr_valid += PSNR((selected_outputs * scaler1[1][idx_start:idx_end][:, None, None, None]).ravel(),
                                    (selected_labels * scaler1[1][idx_start:idx_end][:, None, None, None]).ravel())
                    ssim_valid += ssim((selected_outputs * scaler1[1][idx_start:idx_end][:, None, None, None]), 
                                    (selected_labels * scaler1[1][idx_start:idx_end][:, None, None, None]), 
                                    data_range=4500, size_average=True)
                    if config.classify:
                        clf_logits = clf_logits.argmax(-1)
                        acc_clf_valid += (clf_logits == cls).sum().item() / len(cls)
                    if verbose:
                        loop_valid.set_description(f'Validation {epoch}')
                        loop_valid.set_postfix(loss=loss.item())
                    

            avg_train_loss.append(losses_train / len(train_dataloader))
            avg_valid_loss.append(losses_valid / len(test_dataloader))
            avg_train_psnr.append(psnr_train / len(train_dataloader))
            avg_valid_psnr.append(psnr_valid / len(test_dataloader))
            avg_train_ssim.append(ssim_train / len(train_dataloader))
            avg_valid_ssim.append(ssim_valid / len(test_dataloader))
            time_per_epoch.append(time.time() - epoch_time)
            avg_train_clf_loss.append(losses_clf_train / len(train_dataloader))
            avg_valid_clf_loss.append(losses_clf_valid / len(test_dataloader))
            avg_train_gen_loss.append(losses_gen_train / len(train_dataloader))
            avg_valid_gen_loss.append(losses_gen_valid / len(test_dataloader))
            avg_train_clf_acc.append(acc_clf_train / len(train_dataloader))
            avg_valid_clf_acc.append(acc_clf_valid / len(test_dataloader))

            # warmup and scheduler
            if scheduler is not None:
                if warmup is not None:
                    with warmup.dampening():
                        if config.scheduler == "reduce_on_plateau":
                            scheduler.step(avg_valid_loss[-1])
                        else:
                            scheduler.step()
                else:
                    if config.scheduler == "reduce_on_plateau":
                        scheduler.step(avg_valid_loss[-1])
                    else:
                        scheduler.step()
            elif warmup is not None:
                with warmup.dampening():
                    pass
            
            loop_epoch.set_description(f'Epoch {epoch}')
            loop_epoch.set_postfix(avg_valid_loss=avg_valid_loss[-1])
            
            if verbose:
                print("Epoch time: {:.2f} s".format(time.time() - epoch_time))
                print("Total time elapsed: {:.2f} s".format(time.time() - total_time))
                print("---------------------------------------")
            
            if config.wandb_log:
                wandb.log({"avg_train_loss": avg_train_loss[-1], 
                        "avg_valid_loss": avg_valid_loss[-1], 
                        "avg_train_psnr": avg_train_psnr[-1], 
                        "avg_valid_psnr": avg_valid_psnr[-1], 
                        "avg_train_ssim": avg_train_ssim[-1], 
                        "avg_valid_ssim": avg_valid_ssim[-1], 
                        "time_per_epoch": time_per_epoch[-1],
                        "gpu_memory_used": gpu_memory_used, 
                        "epoch": epoch, 
                        "lr_epoch": lr_epoch[-1], 
                        "avg_train_clf_loss": avg_train_clf_loss[-1], 
                        "avg_valid_clf_loss": avg_valid_clf_loss[-1], 
                        "avg_train_gen_loss": avg_train_gen_loss[-1], 
                        "avg_valid_gen_loss": avg_valid_gen_loss[-1], 
                        "avg_train_clf_acc": avg_train_clf_acc[-1], 
                        "avg_valid_clf_acc": avg_valid_clf_acc[-1]})
            
            if plot:
                ax.cla()
                ax.plot(np.arange(1, epoch+2), avg_train_loss,'b', label='Training Loss')
                ax.plot(np.arange(1, epoch+2), avg_valid_loss, 'orange', label='Validation Loss')
                ax.legend()
                ax.set_title("Loss Curve")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Avg Loss")
                f.canvas.draw()

            if config.patience is not None:
                early_stopping(avg_valid_loss[-1], model, optim=optim, scheduler=scheduler)

                if early_stopping.early_stop:
                    print("Early stopping")
                    if config.wandb_log:
                        wandb.log({"avg_train_loss": avg_train_loss[epoch-config.patience], 
                                "avg_valid_loss": avg_valid_loss[epoch-config.patience], 
                                "avg_train_psnr": avg_train_psnr[epoch-config.patience], 
                                "avg_valid_psnr": avg_valid_psnr[epoch-config.patience], 
                                "avg_train_ssim": avg_train_ssim[epoch-config.patience], 
                                "avg_valid_ssim": avg_valid_ssim[epoch-config.patience], 
                                "avg_train_clf_loss": avg_train_clf_loss[epoch-config.patience], 
                                "avg_valid_clf_loss": avg_valid_clf_loss[epoch-config.patience], 
                                "avg_train_gen_loss": avg_train_gen_loss[epoch-config.patience], 
                                "avg_valid_gen_loss": avg_valid_gen_loss[epoch-config.patience], 
                                "avg_train_clf_acc": avg_train_clf_acc[epoch-config.patience], 
                                "avg_valid_clf_acc": avg_valid_clf_acc[epoch-config.patience]})
                    break
        
        if config.patience is not None:
            model.load_state_dict(torch.load(checkpoint)['model'])
            optim.load_state_dict(torch.load(checkpoint)['optim'])
            if scheduler is not None:
                scheduler.load_state_dict(torch.load(checkpoint)['scheduler'])
    
    except KeyboardInterrupt:
        print("Stopped training, returning to last checkpoint...")
        model.load_state_dict(torch.load(checkpoint)['model'])
        optim.load_state_dict(torch.load(checkpoint)['optim'])
        if scheduler is not None:
            scheduler.load_state_dict(torch.load(checkpoint)['scheduler'])

    return model, avg_train_loss, avg_valid_loss, time_per_epoch

def run_velup(model, optim, warmup, scheduler, loss_fn, train_dataloader, test_dataloader, scaler1, config, 
               plot=False, f=None, ax=None, verbose=True):
    epochs = config.epoch
    device = config.device
    total_time = time.time()
    avg_train_loss = []
    avg_valid_loss = []
    avg_train_psnr = []
    avg_valid_psnr = []
    avg_train_ssim = []
    avg_valid_ssim = []
    time_per_epoch = []
    lr_epoch = []
    if config.patience is not None:
        checkpoint = os.path.join(config.parent_dir, str(os.getpid())+"checkpoint.pt")
        early_stopping = EarlyStopping(patience=config.patience, verbose=False, path=checkpoint)
    
    start_epoch = get_previous_epoch_count(config) if hasattr(config, 'cont_dir') and config.cont_dir else 0
    loop_epoch = tqdm(range(start_epoch, epochs))
    for epoch in loop_epoch:
        epoch_time = time.time()
        lr_epoch.append(get_lr(optim))
        model.train()
        # setup loop with TQDM and dataloader
        if verbose:
            loop_train = tqdm(train_dataloader, leave=True, position=0)
        else:
            loop_train = train_dataloader
        losses_train = 0
        psnr_train = 0
        ssim_train = 0
        for i, batch in enumerate(loop_train):
            # initialize calculated gradients (from prev step)
            optim.zero_grad()

            # pull all tensor batches required for training
            inputs = batch['input'].to(config.device)
            labels = batch['label'].to(config.device)

            # process
            x_tilde = model(inputs.unsqueeze(1))
            loss = loss_fn(x_tilde, labels.unsqueeze(1))

            loss.backward()

            # update parameters
            optim.step()
            
            # calculate metrics
            losses_train += loss.item()
            with torch.no_grad():
#                 outputs = _to_sequence(x_tilde, config, inv=True)
                outputs = x_tilde.squeeze(1)
                selected_outputs = (outputs.unsqueeze(1) / config.scaler2) + config.scaler3
                selected_labels = (labels.unsqueeze(1) / config.scaler2) + config.scaler3
                idx_start = i*config.batch_size
                idx_end = (i+1)*config.batch_size
                psnr_train += PSNR((selected_outputs * scaler1[0][idx_start:idx_end][:, None, None, None]).ravel(),
                                   (selected_labels * scaler1[0][idx_start:idx_end][:, None, None, None]).ravel())
                ssim_train += ssim((selected_outputs * scaler1[0][idx_start:idx_end][:, None, None, None]) + 1, 
                                   (selected_labels * scaler1[0][idx_start:idx_end][:, None, None, None]) + 1, 
                                   data_range=2, size_average=True)
            if verbose:
                loop_train.set_description(f'Epoch {epoch}')
                loop_train.set_postfix(loss=loss.item())
            
            if i == 0:
                nvsmi = nvidia_smi.getInstance()
                gpu_memory_used = nvsmi.DeviceQuery('memory.used')['gpu'][0]['fb_memory_usage']['used']
                        
        model.eval()
        if verbose:
            loop_valid = tqdm(test_dataloader, leave=True, position=0)
        else:
            loop_valid = test_dataloader
        losses_valid = 0
        psnr_valid = 0
        ssim_valid = 0
        with torch.no_grad():
            for i, batch in enumerate(loop_valid):
                # pull all tensor batches required for training
                inputs = batch['input'].to(config.device)
                labels = batch['label'].to(config.device)

                # process
                x_tilde = model(inputs.unsqueeze(1))
                loss = loss_fn(x_tilde, labels.unsqueeze(1))
                
                # calculate metrics
                losses_valid += loss.item()
#                 outputs = _to_sequence(x_tilde, config, inv=True)
                outputs = x_tilde.squeeze(1)
                selected_outputs = (outputs.unsqueeze(1) / config.scaler2) + config.scaler3
                selected_labels = (labels.unsqueeze(1) / config.scaler2) + config.scaler3
                idx_start = i*config.batch_size
                idx_end = (i+1)*config.batch_size
                psnr_valid += PSNR((selected_outputs * scaler1[1][idx_start:idx_end][:, None, None, None]).ravel(),
                                   (selected_labels * scaler1[1][idx_start:idx_end][:, None, None, None]).ravel())
                ssim_valid += ssim((selected_outputs * scaler1[1][idx_start:idx_end][:, None, None, None]) + 1, 
                                   (selected_labels * scaler1[1][idx_start:idx_end][:, None, None, None]) + 1, 
                                   data_range=2, size_average=True)
                if verbose:
                    loop_valid.set_description(f'Validation {epoch}')
                    loop_valid.set_postfix(loss=loss.item())
                

        avg_train_loss.append(losses_train / len(train_dataloader))
        avg_valid_loss.append(losses_valid / len(test_dataloader))
        avg_train_psnr.append(psnr_train / len(train_dataloader))
        avg_valid_psnr.append(psnr_valid / len(test_dataloader))
        avg_train_ssim.append(ssim_train / len(train_dataloader))
        avg_valid_ssim.append(ssim_valid / len(test_dataloader))
        time_per_epoch.append(time.time() - epoch_time)

        # warmup and scheduler
        if scheduler is not None:
            if warmup is not None:
                with warmup.dampening():
                    if config.scheduler == "reduce_on_plateau":
                        scheduler.step(avg_valid_loss[-1])
                    else:
                        scheduler.step()
            else:
                if config.scheduler == "reduce_on_plateau":
                    scheduler.step(avg_valid_loss[-1])
                else:
                    scheduler.step()
        elif warmup is not None:
            with warmup.dampening():
                pass
        
        loop_epoch.set_description(f'Epoch {epoch}')
        loop_epoch.set_postfix(avg_valid_loss=avg_valid_loss[-1])
        
        if verbose:
            print("Epoch time: {:.2f} s".format(time.time() - epoch_time))
            print("Total time elapsed: {:.2f} s".format(time.time() - total_time))
            print("---------------------------------------")
        
        if config.wandb_log:
            wandb.log({"avg_train_loss": avg_train_loss[-1], 
                    "avg_valid_loss": avg_valid_loss[-1], 
                    "avg_train_psnr": avg_train_psnr[-1], 
                    "avg_valid_psnr": avg_valid_psnr[-1], 
                    "avg_train_ssim": avg_train_ssim[-1], 
                    "avg_valid_ssim": avg_valid_ssim[-1], 
                    "time_per_epoch": time_per_epoch[-1],
                    "gpu_memory_used": gpu_memory_used, 
                    "epoch": epoch, 
                    "lr_epoch": lr_epoch[-1]})
        
        if plot:
            ax.cla()
            ax.plot(np.arange(1, epoch+2), avg_train_loss,'b', label='Training Loss')
            ax.plot(np.arange(1, epoch+2), avg_valid_loss, 'orange', label='Validation Loss')
            ax.legend()
            ax.set_title("Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Avg Loss")
            f.canvas.draw()

        if config.patience is not None:
            early_stopping(-avg_valid_ssim[-1], model)

            if early_stopping.early_stop:
                print("Early stopping")
                if config.wandb_log:
                    wandb.log({"avg_train_loss": avg_train_loss[epoch-config.patience], 
                            "avg_valid_loss": avg_valid_loss[epoch-config.patience], 
                            "avg_train_psnr": avg_train_psnr[epoch-config.patience], 
                            "avg_valid_psnr": avg_valid_psnr[epoch-config.patience], 
                            "avg_train_ssim": avg_train_ssim[epoch-config.patience], 
                            "avg_valid_ssim": avg_valid_ssim[epoch-config.patience]})
                break
    
    if config.patience is not None:
        model.load_state_dict(torch.load(checkpoint))

    return model, avg_train_loss, avg_valid_loss, time_per_epoch