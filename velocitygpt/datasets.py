import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
import multiprocessing
import pylops

class ElasticGPTDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for ElasticGPT."""

    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.data.items()}
    def __len__(self):
        return len(self.data['input'])

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

def resize_image2(image, image_size=None, inv=False, orig_image_size=None):
    if not inv and image_size is not None:
        image = resize(image, image_size, order=0, anti_aliasing=False, preserve_range=True)
    elif inv and orig_image_size is not None:
        image = resize(image, orig_image_size, order=0, anti_aliasing=False, preserve_range=True)
    
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