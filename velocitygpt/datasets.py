import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from skimage.transform import resize
import multiprocessing
import pylops
from .NpyDataset import NpyDataset
import os
from torchvision.transforms.functional import gaussian_blur
from copy import deepcopy

class ElasticGPTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for ElasticGPT with flexible transformations."""

    def __init__(self, data=None, transform=None, config=None, train=True):
        def create_global_position(x, y, z, comp_rate=4, mode='full', savedir='glob_pos.npy'):
            x, z = x//comp_rate, z//comp_rate
            if mode == 'full':
                glob_pos = torch.arange(z * x).reshape(z, x)
            elif mode == 'depth':
                glob_pos = torch.arange(z).repeat_interleave(x).reshape(z, x)
            glob_pos = glob_pos.repeat_interleave(comp_rate, dim=0).repeat_interleave(comp_rate, dim=1)
            glob_pos = glob_pos.unsqueeze(0).repeat(y, 1, 1).numpy()
            
            np.save(savedir, glob_pos)

        self.data = data
        self.config = config
        if self.config is not None and self.data is None:
            if train:
                range = {'y': (0, config.train_prop*config.prop)} 
            else:
                range = {'y': (config.train_prop*config.prop, 1*config.prop)}
            dataset_path = [x[:2] for x in config.dataset_path]
            paths = [{'data': dp, 'label': lp, 'order': ('y', 'z', 'x'), 'range': range} for [dp, lp] in dataset_path]
            self.data = NpyDataset(paths=paths,
                                   norm=0,
                                   window_w=config.image_size[0],
                                   window_h=config.image_size[1], 
                                   stride_w=config.stride[0], 
                                   stride_h=config.stride[1],
                                   mode='windowed', 
                                   line_mode='xline')
            if config.use_dip:
                x_glob_pos = self.data.ranges[0][1] - self.data.ranges[0][0]
                y_glob_pos = self.data.ranges[0][3] - self.data.ranges[0][2]
                z_glob_pos = self.data.ranges[0][5] - self.data.ranges[0][4]
                create_global_position(x_glob_pos, y_glob_pos, z_glob_pos, 
                                       comp_rate=config.image_size[1]//config.latent_dim[1], 
                                       mode=config.glob_pos_mode, 
                                       savedir=os.path.join(config.parent_dir, 'glob_pos.npy'))
                self.glob_pos = NpyDataset(paths=[{'data': os.path.join(config.parent_dir, 'glob_pos.npy'), 'order': ('y', 'z', 'x')}],
                                            norm=0,
                                            window_w=config.image_size[0],
                                            window_h=config.image_size[1], 
                                            stride_w=config.stride[0], 
                                            stride_h=config.stride[1],
                                            mode='windowed', 
                                            line_mode='xline', 
                                            mmap_mode=None)
            else:
                self.glob_pos = None

        self.transform = transform  # Accept a transform or None

    def __getitem__(self, idx):
        if hasattr(self, 'config'):
            if self.config.dataset_type in ["fld2", "syn2"]:
                sample = {}
                sample['input'], sample['label'] = self.data[idx]
                sample['input'] = torch.tensor(sample['input'].T).float()
                sample['label'] = torch.tensor(sample['label'].T).float()
                if self.glob_pos is not None:
                    sample['dip_seq'] = torch.tensor(self.glob_pos[idx][0]).long()
                    h, w =  self.config.image_size[1]//self.config.latent_dim[1], self.config.image_size[0]//self.config.latent_dim[0]
                    sample['dip_seq'] = sample['dip_seq'][::h, ::w].flatten()
            else:
                sample = {key: val[idx].clone().detach() for key, val in self.data.items()}

            sample['input'] = {'tensor': sample['input']}
            sample['label'] = {'tensor': sample['label']}

            # Apply the transform if specified
            if self.transform:
                sample['input'], sample['label'], sample['label2'] = self.transform(sample['input'], sample['label'])
        else:
            sample = {key: val[idx].clone().detach() for key, val in self.data.items()}

            # Apply the transform if specified
            if self.transform:
                sample['input'] = self.transform(sample['input'].unsqueeze(0)).squeeze(0)
        
        return sample

    def __len__(self):
        if hasattr(self, 'config'):
            if self.config.dataset_type in ["fld2", "syn2"]:
                return len(self.data)
            else:
                return len(self.data['input'])
        else:
            return len(self.data['input'])
        
    def denormalize(self, tensor_dict):
        """Revert the normalization for visualization."""
        denorm_mode = self.config.norm_mode[::-1]
        for nm in denorm_mode:
            if nm == 'max':
                tensor_dict['tensor'] *= tensor_dict['max']
            elif nm == 'mean_std':
                tensor_dict['tensor'] = tensor_dict['tensor'] * tensor_dict['std'] + tensor_dict['mean']
        return tensor_dict['tensor']

class Normalization:
    def __init__(self, config):
        self.norm_mode = config.norm_mode
        self.norm_level = config.norm_level
        self.norm_const = config.norm_const
        self.scaler2 = config.scaler2
        self.scaler3 = config.scaler3
    def __call__(self, tensor1, tensor2, tensor3={}):
        for i, nm in enumerate(self.norm_mode):
            if nm == "max":
                for j, td in enumerate([tensor1, tensor2, tensor3]):
                    if td:
                        if self.norm_level == "trace":
                            max_val = td['tensor'].max(dim=1, keepdim=True).values
                        elif self.norm_level == "sample":
                            max_val = td['tensor'].max().reshape(-1, 1)
                        else:
                            max_val = torch.tensor(self.norm_const[j]).reshape(-1, 1)
                        td['max'] = max_val
                        td['tensor'] /= (td['max'] + 1e-8)
            if nm == "mean_std":
                for j, td in enumerate([tensor1, tensor2, tensor3]):
                    if td:
                        if self.norm_level == "sample":
                            mean_val = td['tensor'].mean(dim=[0, 1], keepdim=True)
                            std_val = td['tensor'].std(dim=[0, 1], keepdim=True)
                        elif self.norm_level == "trace":
                            mean_val = td['tensor'].mean(dim=[1], keepdim=True)
                            std_val = td['tensor'].std(dim=[1], keepdim=True)
                        else:
                            mean_val = torch.tensor(self.scaler3[j]).reshape(-1, 1)
                            std_val = torch.tensor(self.scaler2[j]).reshape(-1, 1)
                        td['mean'] = mean_val
                        td['std'] = std_val
                        td['tensor'] = (td['tensor'] - td['mean']) / (td['std'] + 1e-8)  # Add epsilon to avoid division by zero
        return tensor1, tensor2, tensor3  
    
class Flip:
    def __call__(self, tensor1, tensor2, tensor3={}):
        if torch.rand(1).item() < 0.5:
            return tensor1, tensor2, tensor3 
        else:
            for td in [tensor1, tensor2, tensor3]:
                if td:
                    td['tensor'] = torch.flip(td['tensor'], dims=(0,))
            return tensor1, tensor2, tensor3
        
def apply_gaussian_filter(tensor, kernel_size, sigma):
    if sigma > 0:
        tensor = gaussian_blur(tensor.unsqueeze(0),
                            kernel_size=[kernel_size, kernel_size],
                            sigma=[sigma, sigma]).squeeze(0)
    
    return tensor
        
class GaussianFilter:
    def __init__(self, config):
        self.sigma = config.transform_gaussian_sigma
        self.use_init_prob = config.use_init_prob

    def __call__(self, tensor1, tensor2, tensor3={}, sigma=None):
        if sigma is None:
            if len(self.sigma) > 1: # Pick from range
                sigma = torch.empty(1).uniform_(*self.sigma).item()
            else: # Binary
                sigma = self.sigma[0] if torch.rand(1).item() < 0.5 else 0
        kernel_size = round(4 * sigma) # scipy.ndimage default
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size # Make sure odd
        if self.use_init_prob > 0 and tensor3:
            tensor3['tensor'] = apply_gaussian_filter(tensor3['tensor'], kernel_size, sigma)
        else: # Apply to input
            tensor1['tensor'] = apply_gaussian_filter(tensor1['tensor'], kernel_size, sigma)

        return tensor1, tensor2, tensor3

class DualTransform:
    def __init__(self, transforms, config):
        self.transforms = transforms
        self.use_init_prob = config.use_init_prob
    
    def __call__(self, tensor1, tensor2):
        tensor3 = deepcopy(tensor1) if self.use_init_prob > 0 else {}
        for t in self.transforms:
            tensor1, tensor2, tensor3 = t(tensor1, tensor2, tensor3)
        return tensor1, tensor2, tensor3

# Function to create a list of transforms based on arguments
def create_transforms(args):
    transform_list = []
    
    # Add Gaussian blur if specified
    if sum(args.transform_gaussian_sigma) > 0:
        transform_list.append(GaussianFilter(args))
        
    if args.aug_flip:
        transform_list.append(Flip())

    if args.dataset_type in ["fld2", "syn2"]:
        transform_list.append(Normalization(args))
    
    if transform_list:
        return DualTransform(transform_list, args)
    else:
        return None  # No transform if the list is empty

def SeisPWD(in_data, w1=5, w2=5, dz_in=1, dx_in=1, format="angle"):
    # dip estimation by Plane Wave Destruction
    # see http://sepwww.stanford.edu/data/media/public/sep//prof/pvi.pdf Chapter 4.

    d = np.copy(in_data)
    n1, n2 = in_data.shape

    # format = get(param,"format","angle") # output format: "angle" (in degrees wrt vertical) or "dip" (in y samples over x samples)

    pp = np.zeros((n1, n2))
    dx = wavekill(1., pp, n1, n2, d)
    pp = np.ones((n1, n2))
    dt = wavekill(0., pp, n1, n2, d)
    dtdx = dt * dx
    dxdx = dx * dx
    dtdt = dt * dt

    for i2 in range(n2):
        dtdt[:, i2] = triangle(w1, n1, dtdt[:, i2])
        dxdx[:, i2] = triangle(w1, n1, dxdx[:, i2])
        dtdx[:, i2] = triangle(w1, n1, dtdx[:, i2])

    for i1 in range(n1):
        dtdt[i1, :] = triangle(w2, n2, dtdt[i1, :])
        dxdx[i1, :] = triangle(w2, n2, dxdx[i1, :])
        dtdx[i1, :] = triangle(w2, n2, dtdx[i1, :])

    coh = np.sqrt((dtdx * dtdx) / (dtdt * dxdx))
    pp = -dtdx / dtdt

    coh = np.zeros((n1, n2))
    for i1 in range(n1):
        for i2 in range(n2):
            if abs(dtdt[i1, i2]) > 1e-6:
                coh[i1, i2] = np.sqrt((dtdx[i1, i2] * dtdx[i1, i2]) / (dtdt[i1, i2] * dxdx[i1, i2]))
                pp[i1, i2] = -dtdx[i1, i2] / dtdt[i1, i2]
            else:
                coh[i1, i2] = 0.
                pp[i1, i2] = 0.

    for i2 in range(n2):
        pp[:, i2] = triangle(w1, n1, pp[:, i2])

    for i1 in range(n1):
        pp[i1, :] = triangle(w2, n2, pp[i1, :])

    res = wavekill(1., pp, n1, n2, d)

    if format == "angle":
        pp = np.arctan(pp * dz_in / dx_in) * 180 / np.pi

    return coh, pp, res


def wavekill(aa, bb, n1, n2, uu):
    vv = np.zeros((n1, n2))
    s11 = -aa - bb
    s12 = aa - bb
    s21 = -aa + bb
    s22 = aa + bb
    for i2 in range(n2 - 1):
        for i1 in range(n1 - 1):
            vv[i1, i2] = uu[i1, i2] * s11[i1, i2] + uu[i1, i2 + 1] * s12[i1, i2] + uu[i1 + 1, i2] * s21[i1, i2] + \
                         uu[i1 + 1, i2 + 1] * s22[i1, i2]
    vv[n1 - 1, :] = vv[n1 - 2, :]
    vv[:, n2 - 1] = vv[:, n2 - 2]
    return vv


def triangle(nbox, nd, xx):
    yy = np.zeros(nd)
    pp = boxconv(nbox, nd, xx)
    npp = nbox + nd - 1
    qq = boxconv(nbox, npp, pp)
    nq = nbox + npp - 1
    for i in range(nd):
        yy[i] = qq[i + nbox - 1]
    for i in range(nbox - 1):
        yy[i] = yy[i] + qq[nbox - i]
    for i in range(nbox - 1):
        yy[nd - i - 1] = yy[nd - i - 1] + qq[nd + (nbox - 1) + i]
    return yy


def boxconv(nbox, nx, xx):
    ny = nx + nbox - 1
    yy = np.zeros(ny)
    bb = np.zeros(nx + nbox)
    bb[0] = xx[0]
    for i in range(1, nx):
        bb[i] = bb[i - 1] + xx[i]
    for i in range(nx, ny):
        bb[i] = bb[i - 1]
    for i in range(nbox):
        yy[i] = bb[i]
    for i in range(nbox, ny):
        yy[i] = bb[i] - bb[i - nbox]
    for i in range(ny):
        yy[i] = yy[i] / nbox
    return yy

def patchify(inp, kernel_size, stride, inv=False, orig_size=None):
    if not inv:
        out = inp.unfold(0, kernel_size[0], stride[0]).unfold(1, kernel_size[1], stride[1])
        out = out.reshape(-1, kernel_size[0], kernel_size[1])
        input_ones = torch.ones_like(inp).unfold(0, kernel_size[0], stride[0]).unfold(1, kernel_size[1], stride[1])
        divisor = input_ones.reshape(-1, kernel_size[0], kernel_size[1])

        return out

    elif inv and orig_size is not None:
        out = inp.reshape(-1, kernel_size[0]*kernel_size[1])
        out = F.fold(out.transpose(0, 1), output_size=orig_size, kernel_size=kernel_size, stride=stride)[0]
        divisor = torch.ones_like(inp)
        divisor = divisor.reshape(-1, kernel_size[0]*kernel_size[1])
        divisor = F.fold(divisor.transpose(0, 1), output_size=orig_size, kernel_size=kernel_size, stride=stride)[0]

        return out / divisor

def _get_dip(models, bins, patch_size, return_dip=False):
    bins = np.array(bins)
    dip_seq = []
    if return_dip:
        dip = []
    with multiprocessing.Pool(30) as pool:
        out = pool.starmap(SeisPWD, zip(models))
    pp = [x[1] for x in out]
    for j in range(len(pp)):
        if return_dip:
            dip.append(torch.tensor(pp[j]))
        dip_ = np.digitize(pp[j], bins).T
        dip_ = patchify(torch.tensor(dip_), patch_size, patch_size)
        dip_ = dip_.mode(-1).values.mode(-1).values
        dip_seq.append(dip_)
    dip_seq = torch.stack(dip_seq, dim=0)
    
    if return_dip:
        dip = torch.stack(dip, dim=0)
        return dip_seq, dip
    
    else:
        return dip_seq
    
def pad_input(inp, pad, inv=False, orig_shape=None):
    if not inv:
        inp = F.pad(inp, (pad[0], pad[1], pad[2], pad[3], 0, 0), mode='constant', value=0)
    elif inv and orig_shape is not None:
        ns, no, nt = orig_shape
        inp = inp[:, pad[2]:pad[2]+no, pad[0]:pad[0]+nt]
        
    return inp

def resize_image(image, image_size=None, inv=False, orig_image_size=None):
    if not inv and image_size is not None:
        image = resize(image, image_size, order=5, anti_aliasing=True, preserve_range=True)
    elif inv and orig_image_size is not None:
        image = resize(image, orig_image_size, order=5, anti_aliasing=True, preserve_range=True)
    
    return image

def resize_image2(image, image_size=None, inv=False, orig_image_size=None, anti_aliasing=False, order=0):
    if not inv and image_size is not None:
        image = resize(image, image_size, order=order, anti_aliasing=anti_aliasing, preserve_range=True)
    elif inv and orig_image_size is not None:
        image = resize(image, orig_image_size, order=order, anti_aliasing=anti_aliasing, preserve_range=True)
    
    return image

def make_AI(vp):
    taper = vp < 1.01 * torch.min(vp)
    rho = 1e3*0.3 * vp**0.25
    rho = torch.where(taper, 1000., rho)
    
    AI = vp * rho
    
    return AI

def make_refl(AI):
    refl = F.pad(AI[None, :, :], pad=(0, 1, 0, 0), mode="replicate")[0]
    refl = (refl[..., 1:] - refl[..., :-1]) / (refl[..., 1:] + refl[..., :-1])
    
    return refl

def make_post(AI, wav):
    ns, nx, nz = AI.shape
    m = np.log(AI.transpose(2, 1, 0))

    # dense operator
    PPop_dense = pylops.avo.poststack.PoststackLinearModelling(
        wav / 2, nt0=nz, spatdims=(ns, nx), explicit=True
    )

    # data
    d = (PPop_dense * m.ravel())
    d = d.reshape(nz, nx, ns).transpose(2, 1, 0)
    d = torch.tensor(d).float()
    
    return d