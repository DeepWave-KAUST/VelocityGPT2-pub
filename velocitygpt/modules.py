import torch
from torch import nn
import torch.nn.functional as F
import math
import distributed as dist_fn 
from torch.autograd import Function
from fast_transformers.attention import AttentionLayer, CausalLinearAttention
from fast_transformers.masking import TriangularCausalMask, LengthMask
from linear_attention_transformer.linear_attention_transformer import SelfAttention
from typing import Optional
from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock2(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out
    
class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)
    
vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
# __all__ = ['vq', 'vq_st']
    
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar
    
class ResBlock(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode=config.padding_mode),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding_mode=config.padding_mode),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock2(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock2(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class AdaLN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.SiLU(),
            nn.Linear(d_model*2, d_model*2, bias=True),
        )
        nn.init.zeros_(self.to_gamma_beta[-1].weight)        # “zero-init” trick
    def forward(self, x, cond):                              # cond = pos_i
        gamma, beta = self.to_gamma_beta(cond).chunk(2, -1)
        if len(gamma) != len(x):
            gamma = F.pad(gamma, (0, 0, 0, 0, x.shape[0] - gamma.shape[0], 0), value=0)
            beta = F.pad(beta, (0, 0, 0, 0, x.shape[0] - beta.shape[0], 0), value=0)
        return (1 + gamma) * x + beta

class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_kv_heads (int): number of key/value heads.
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "cache_pos", torch.arange(0, cache_shape[2]), persistent=False
        )
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size

    @property
    def size(self) -> int:
        return self.cache_pos[0].item()

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions. If you wish to update cache values which have
            already been filled, use ``.reset()``, which will reset the cache to the zero-th position.

        Example:
            >>> cache = KVCache(batch_size=2, num_kv_heads=4, max_seq_len=16, head_dim=32, dtype=torch.bfloat16)
            >>> keys, values = torch.ones((2, 4, 8, 32)), torch.ones((2, 4, 8, 32))
            >>> cache.update(keys, values)
            >>> # now positions 0 through 7 are filled
            >>> cache.size
            >>> 8
            >>> keys, values = torch.ones((2, 4, 1, 32)), torch.ones((2, 4, 1, 32))
            >>> cache.update(keys, values)
            >>> # this will fill at position 8
            >>> cache.size
            >>> 9

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.

        Raises:
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.

        Note:
            This function will raise an ``AssertionError`` if the sequence length of ``k_val``
                is longer than the maximum cache sequence length.

        """
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {self.k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )

        assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, self.cache_pos[:seq_len]] = k_val
        v_out[:, :, self.cache_pos[:seq_len]] = v_val

        # forward cache_pos seq_len positions along
        # cache_pos starts at (0, 1, 2, 3, 4, 5, ...)
        # an update of seq_len = 5 tokens brings it to
        # (5, 6, 7, 8, 9, ...)
        # this allows us to track the current position in the cache
        # after the last update in a compile-friendly way without any dynamism
        # e.g. relying on an int size tracker, or re-creating cache_pos every time
        self.cache_pos.add_(seq_len)

        return k_out, v_out

class MultiHeadAttention(nn.Module):
    """Multi-headed attention layer with support for grouped query
    attention (GQA) introduced in https://arxiv.org/abs/2305.13245v1.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    `litgpt.Config <https://github.com/Lightning-AI/litgpt/blob/eda1aaaf391fd689664f95487ab03dc137e213fd/litgpt/config.py>`_).


    ::

        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
        ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
                MHA                    GQA                   MQA
        n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            ``num_heads % num_kv_heads == 0``. For standard MHA set ``num_kv_heads == num_heads``,
            for GQA ``num_kv_heads < num_heads``, and for MQA set ``num_kv_heads == 1``.
        head_dim (int): dimension of each head, calculated by ``embed_dim // num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (Optional[nn.Module]): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        q_norm (Optional[nn.Module]): normalization layer for query, e.g. RMSNorm. For decoding, this is applied
            before updating from kv_cache. This means it will only support token wide normalization and not
            batch or sequence wide normalization.
        k_norm (Optional[nn.Module]): normalization layer for key, must be set if q_norm is.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        is_causal (bool): sets the default mask to causal when no mask is provided
        attn_dropout (float): dropout value passed onto the scaled_dot_product_attention function.
            Default value is 0.0.

    Raises:
        ValueError:
            If ``num_heads % num_kv_heads != 0``, **or**
            if ``embed_dim % num_heads != 0``, **or**
            if ``attn_dropout < 0`` or ``attn_dropout > 1``, **or**
            if q_norm is defined without k_norm or vice versa
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
        bias: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        def _sdpa_call(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor],
            dropout_p: float,
            is_causal: bool,
        ) -> torch.Tensor:
            # shape: [b, 1, s, s]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask is not None:
                mask = mask[:, None, :, :]

            # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
            return nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal
        )
        self._attention_call = _sdpa_call

        # this flag indicates whether to update the kv-cache during forward
        # passes. when disabled, we can have the cache setup but still
        # perform normal forward passes
        self.cache_enabled = False

    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        """Setup key value caches for attention calculation. If called
        after kv_cache is already setup, this will be skipped.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            max_seq_len (int): maximum sequence length model will be run with.
        """
        # Don't overwrite user defined kv_cache from init
        if self.kv_cache is not None:
            print(
                "Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping."
            )
        else:
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.cache_enabled = True

    def reset_cache(self):
        """Reset the key value caches."""
        if self.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )
        self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        bias: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y. Optional only with kv_cache enabled.
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.decoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Raises:
            ValueError: If no ``y`` input and ``kv_cache`` is not enabled.

        Returns:
            torch.Tensor: output tensor with attention applied

        Notation used for tensor shapes:
            - b: batch size
            - s_x: sequence length for x
            - s_y: sequence length for y
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
        """
        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, d = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)
        if bias is not None:
            if isinstance(bias, RotaryEmbedding):
                if bias.freqs_for == "pixel":
                    # Pad if s_x is not divisible by latent_dim
                    pad_len = bias.latent_dim[0] - (s_x % bias.latent_dim[0]) if (s_x % bias.latent_dim[0]) != 0 else 0
                    q = F.pad(q, (0, 0, 0, pad_len), value=0)
                    q = q.view(b, bias.latent_dim[0], -1, d)
                    # Apply 2d RoPE
                    freqs = bias.get_axial_freqs(*q.shape[1:-1])
                    q = apply_rotary_emb(freqs, q)
                    # Remove padding if applied
                    q = q.flatten(1, 2)[:, :s_x, :]

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Apply positional embeddings
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # [b, n_h, s_x, h_d]
        q = q.transpose(1, 2)
        if bias is not None:
            if isinstance(bias, RotaryEmbedding):
                if bias.freqs_for == "lang":
                    q = bias.rotate_queries_or_keys(q)

        # Normalize q
        if self.q_norm is not None:
            q = self.q_norm(q)

        if y is None:
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # Update k and v shape, positional embeddings, and normalization

            # k,v shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)
            v = self.v_proj(y)

            if bias is not None:
                if isinstance(bias, RotaryEmbedding):
                    if bias.freqs_for == "pixel":
                        # Pad if s_x is not divisible by latent_dim
                        pad_len = bias.latent_dim[0] - (s_x % bias.latent_dim[0]) if (s_x % bias.latent_dim[0]) != 0 else 0
                        k = F.pad(k, (0, 0, 0, pad_len), value=0)
                        k = k.view(b, bias.latent_dim[0], -1, d)
                        # Apply 2d RoPE
                        freqs = bias.get_axial_freqs(*k.shape[1:-1])
                        k = apply_rotary_emb(freqs, k)
                        # Remove padding if applied
                        k = k.flatten(1, 2)[:, :s_x, :]

            # Apply positional embeddings
            # k,v shape: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)
            v = v.view(b, s_y, -1, self.head_dim)
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # k,v shape: [b, n_kv, s_y, h_d]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if bias is not None:
                if isinstance(bias, RotaryEmbedding):
                    if bias.freqs_for == "lang":
                        k = bias.rotate_queries_or_keys(k)

            # Normalize k
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Update key-value cache
            if self.kv_cache is not None and self.cache_enabled:
                k, v = self.kv_cache.update(k, v)

        # If needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        # k,v shape: [b, n_kv, s, h_d] -> [b, n_h, s, h_d]
        if self.num_heads != self.num_kv_heads:
            expand_shape = (b, self.num_kv_heads, q_per_kv, -1, self.head_dim)
            k = k.unsqueeze(2).expand(expand_shape).flatten(1, 2)
            v = v.unsqueeze(2).expand(expand_shape).flatten(1, 2)

        # if bias is not None:
        #     if isinstance(bias, RotaryEmbedding):
        #         if bias.freqs_for == "pixel":
        #             # Pad if s_x is not divisible by latent_dim
        #             pad_len = bias.latent_dim[0] - (s_x % bias.latent_dim[0]) if (s_x % bias.latent_dim[0]) != 0 else 0
        #             q = F.pad(q, (0, 0, 0, pad_len), value=0)
        #             k = F.pad(k, (0, 0, 0, pad_len), value=0)
        #             q = q.view(b, self.num_kv_heads * q_per_kv, bias.latent_dim[0], -1, self.head_dim)
        #             k = k.view(b, self.num_kv_heads * q_per_kv, bias.latent_dim[0], -1, self.head_dim)
        #             # Apply 2d RoPE
        #             freqs = bias.get_axial_freqs(*q.shape[2:-1])
        #             q = apply_rotary_emb(freqs, q)
        #             k = apply_rotary_emb(freqs, k)
        #             # Remove padding if applied
        #             q = q.flatten(2, 3)[:, :, :s_x, :]
        #             k = k.flatten(2, 3)[:, :, :s_x, :]
        
        output = self._attention_call(
            q,
            k,
            v,
            # mask=F.pad(mask, (0, k.shape[-2]- mask.shape[-1], 0, 0), value=float("-inf") if mask.dtype != torch.bool else False),
            mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, config, drop_path=0.0):
        super(Block, self).__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        if config.attn_type == "default":
            self.attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=embed_dim // num_heads,
                q_proj=nn.Linear(embed_dim, embed_dim, bias=True),
                k_proj=nn.Linear(embed_dim, embed_dim, bias=True),
                v_proj=nn.Linear(embed_dim, embed_dim, bias=True),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=True)
            )
        elif config.attn_type == "linear":
            self.attn = AttentionLayer(CausalLinearAttention(embed_dim), embed_dim, num_heads)
        elif config.attn_type == "linear2":
            self.attn = SelfAttention(embed_dim, num_heads, causal=True)
        else:
            raise NotImplementedError
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.attn_type = config.attn_type
        
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

        if config.position_embedding_type == "alibi":
            attn_heads = config.num_attention_heads
            if config.fixed_slopes:
                self.slopes = nn.Parameter(torch.Tensor(get_slopes(attn_heads)), requires_grad=False)
            else:
                self.slopes = nn.Parameter(torch.empty(attn_heads), requires_grad=True)
                nn.init.normal_(self.slopes, -2, 1)
        elif config.position_embedding_type == "rope_2d":
            self.bias = RotaryEmbedding(
                dim = config.position_embedding_dim,
                freqs_for = 'pixel',
                max_freq = config.position_embedding_max_freq
            )
            self.bias.latent_dim = config.latent_dim
        elif config.position_embedding_type == "rope_1d":
            self.bias = RotaryEmbedding(
                dim = config.position_embedding_dim,
            )
        else:
            self.bias = None

        if config.adaln_glob_pos:
            self.adaln_glob_pos = AdaLN(embed_dim)
            
        self.position_embedding_type = config.position_embedding_type
        self.num_heads = num_heads
        self.cond_length = config.max_position_embeddings - config.max_length
        self.unmask_condition = config.unmask_condition
        self.max_position_embeddings = config.max_position_embeddings

    def enable_kv_cache(self, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.attn.setup_cache(batch_size, self.attn.q_proj.weight.dtype, self.max_position_embeddings)

    def disable_kv_cache(self):
        if hasattr(self.attn, 'kv_cache') and self.attn.kv_cache is not None:
            self.attn.kv_cache = None
            self.attn.cache_enabled = False

    def forward(self, x, pos=None, attn_mask=None):
        if self.attn_type == "default":
            if attn_mask is None:
                attn_mask = torch.full(
                    (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
                )
                attn_mask = torch.triu(attn_mask, diagonal=1)
                if self.unmask_condition:
                    attn_mask[:self.cond_length, :self.cond_length] = 0
        elif self.attn_type == "linear":
            attn_mask = TriangularCausalMask(len(x), device=x.device)
        
        if self.position_embedding_type == "alibi":
            maxpos = len(x)
            alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos, device=x.device).unsqueeze(0).unsqueeze(0).expand(self.num_heads, -1, -1)
            alibi = alibi.view(self.num_heads, 1, maxpos)
            attn_mask = attn_mask.reshape(1, *attn_mask.shape).repeat(x.shape[1]*self.num_heads, 1, 1)
            attn_mask += alibi.repeat(x.shape[1], attn_mask.shape[1], 1)#.to(x.device)

        x = self.ln_1(x)
        if pos is not None and hasattr(self, 'adaln_glob_pos'):
            x = self.adaln_glob_pos(x, pos)
        if self.attn_type == "default":
            a = self.attn(
                    x.transpose(0, 1), 
                    x.transpose(0, 1),
                    mask=attn_mask,
                    bias=self.bias
                ).transpose(0, 1)
        elif self.attn_type == "linear":
            x = x.transpose(0, 1)
            a = self.attn(x, x, x, 
                          attn_mask=attn_mask, 
                          query_lengths=LengthMask(x.new_full((x.shape[0],), x.shape[1], dtype=torch.int64)), 
                          key_lengths=LengthMask(x.new_full((x.shape[0],), x.shape[1], dtype=torch.int64)))
            a = a.transpose(0, 1)
            x = x.transpose(0, 1)
        elif self.attn_type == "linear2":
            a = self.attn(x.transpose(0, 1))
            a = a.transpose(0, 1)
        x = x + self.drop_path(a)
        m = self.mlp(self.ln_2(x))
        x = x + self.drop_path(m)
        if self.position_embedding_type == "peg":
            N, B, C = x.shape
            
        return x
    
class WDSRBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, wn, act=nn.ReLU(True), res_scale=1):
        super(WDSRBlock, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            wn(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'(drop_prob=...)'