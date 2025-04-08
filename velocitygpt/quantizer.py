from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from torch.cuda.amp import autocast
from einops import rearrange, pack, unpack
import random
from vector_quantize_pytorch import VectorQuantize, LFQ, FSQ

from .modules import *

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    """
    Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
    Code adapted from Jax version in Appendix A.1
    Credits: https://github.com/lucidrains/vector-quantize-pytorch
    """
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = False,
        projection_has_bias: bool = True
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim, bias = projection_has_bias) if has_projections else nn.Identity()

        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        self.allowed_dtypes = allowed_dtypes

    def bound(self, z, eps: float = 1e-3):
        """ Bound `z`, an array of shape (..., d). """
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """ Quantizes z, returns quantized zhat, same shape as z. """
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """ Converts a `code` to an index in the codebook. """
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """ Converts indices to indices at each level, perhaps needed for a transformer with factorized embeddings """
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """ Inverse of `codes_to_indices`. """

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    @autocast(enabled = False)
    def forward(self, z):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """

        orig_dtype = z.dtype
        is_img_or_video = z.ndim >= 4
        need_move_channel_last = is_img_or_video or self.channel_first

        # standardize image or video into (batch, seq, dimension)

        if need_move_channel_last:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # make sure allowed dtype before quantizing

        if z.dtype not in self.allowed_dtypes:
            z = z.float()

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        # cast codes back to original dtype

        if codes.dtype != orig_dtype:
            codes = codes.type(orig_dtype)

        # project out

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if need_move_channel_last:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # return quantized output and indices

        return out, indices

class VectorQuantizedVAE(nn.Module):
    def __init__(self, config):
        input_dim = config.input_dim
        dim = config.dim
        K = config.K
        intermediate_dim = config.intermediate_dim
        self.quantizer = config.quantizer
        super().__init__()
        if config.n_layer == 2:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, dim, 4, 2, 1, padding_mode=config.padding_mode),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, intermediate_dim, 4, 2, 1, padding_mode=config.padding_mode),
                ResBlock(intermediate_dim, config),
                ResBlock(intermediate_dim, config),
            )

            self.decoder = nn.Sequential(
                ResBlock(intermediate_dim, config),
                ResBlock(intermediate_dim, config),
                nn.ReLU(True),
                nn.ConvTranspose2d(intermediate_dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
                nn.Tanh()
            )
        elif config.n_layer == 3:
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, dim, 4, 2, 1, padding_mode=config.padding_mode),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 4, 2, 1, padding_mode=config.padding_mode),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, intermediate_dim, 4, 2, 1, padding_mode=config.padding_mode),
                ResBlock(intermediate_dim, config),
                ResBlock(intermediate_dim, config),
            )

            self.decoder = nn.Sequential(
                ResBlock(intermediate_dim, config),
                ResBlock(intermediate_dim, config),
                nn.ReLU(True),
                nn.ConvTranspose2d(intermediate_dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
                nn.Tanh()
            )

        self.codebook = VQEmbedding(K, intermediate_dim)

        if config.quantizer == "vq":
            self.codebook = VectorQuantize(
                                dim=intermediate_dim,
                                codebook_size=config.K,     # codebook size
                                decay=0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                                commitment_weight=config.beta,
                                accept_image_fmap=True   # the weight on the commitment loss
                            )
        elif config.quantizer == "lfq":
            self.codebook = LFQ(
                                codebook_size=config.K,      # codebook size, must be a power of 2
                                dim=intermediate_dim,                   # this is the input feature dimension, defaults to log2(codebook_size) if not defined
                                entropy_loss_weight=config.quantizer_entropy_weight,  # how much weight to place on entropy loss
                                diversity_gamma=1,       # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
                                commitment_loss_weight=config.beta,
                                soft_clamp_input_value=None,
                                spherical=False
                            )
        elif config.quantizer == "fsq":
            self.codebook = FSQ(levels=config.quantizer_levels,
                                dim=intermediate_dim)
        else:
            raise NotImplementedError

        self.apply(weights_init)
        
        self.add_input_latent_noise = config.add_input_latent_noise
        self.input_noise_factor = config.input_noise_factor
        self.latent_noise_factor = config.latent_noise_factor

    def encode(self, x):
        z_e_x = self.encoder(x)
        if self.quantizer == 'fsq':
            _, latents = self.codebook(z_e_x)
        else:
            _, latents, _ = self.codebook(z_e_x)            
        return latents

    def decode(self, latents):
        if self.quantizer == 'fsq':
            z_q_x = self.codebook.indices_to_codes(latents)
        else:
            z_q_x = self.codebook.get_codes_from_indices(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde
    
    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def forward(self, x, return_latents=False):
        if self.add_input_latent_noise:
            x = self.add_noise(x, self.input_noise_factor)
        z_e_x = self.encoder(x)
        if self.quantizer == 'fsq':
            z_q_x, latents = self.codebook(z_e_x)
            aux_loss = 0
        else:
            z_q_x, latents, aux_loss = self.codebook(z_e_x)
        if self.add_input_latent_noise:
            z_q_x = self.add_noise(z_q_x, self.latent_noise_factor)
            z_q_x = torch.clamp(z_q_x, -1, 1)
        x_tilde = self.decoder(z_q_x)
        if return_latents:
            return x_tilde, z_e_x, aux_loss, latents
        else:
            return x_tilde, z_e_x, aux_loss

class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        in_channel=config.input_dim
        channel=config.dim
        n_res_block=config.n_res_block
        n_res_channel=config.intermediate_dim
        embed_dim=config.intermediate_dim
        n_embed=config.K
        decay=0.99

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
