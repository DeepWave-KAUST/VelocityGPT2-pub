import torch
from torch import nn
import math
from .modules import *

class GPT2(nn.Module):
    def __init__(self, config):
        super(GPT2, self).__init__()

        self.hidden_size = config.hidden_size
        self.latent_dim = config.latent_dim
        self.classify = config.classify
        self.cls_token = config.cls_token

        # start of sequence token
        if not self.cls_token:
            self.sos = torch.nn.Parameter(torch.zeros(config.hidden_size))
            nn.init.normal_(self.sos)
        elif self.cls_token:
            self.cls_token_embeddings = nn.Embedding(config.num_classes, config.hidden_size)

        self.token_embeddings = nn.Embedding(config.vocab_size+math.ceil(config.well_cond_prob or config.use_init_prob), 
                                             config.hidden_size//config.n_concat_token)
        
        if config.position_embedding_type not in ["alibi", "none"]:
            if config.add_pos_first:
                self.position_embeddings = nn.Embedding(config.max_position_embeddings, 
                                                        config.hidden_size//config.n_concat_token)
            else:
                self.position_embeddings = nn.Embedding(config.max_position_embeddings//config.n_concat_token, 
                                                        config.hidden_size)

            if config.double_pos:
                self.position_embeddings2 = nn.Embedding(config.max_position_embeddings//config.n_concat_token, 
                                                         config.hidden_size)
            
        if config.well_cond_prob > 0:
            self.well_position_embeddings = nn.Embedding(config.image_size[0]+1, config.hidden_size)
            
        if config.use_dip:
            self.use_dip_embeddings = nn.Embedding(2, config.hidden_size)
            self.dip_embeddings = nn.Embedding(len(config.dip_bins)+1, config.hidden_size)
            
        if config.vqvae_refl_dir is not None:
            self.use_refl_embeddings = nn.Embedding(2, config.hidden_size)
            self.refl_embeddings = nn.Embedding(config.refl_vocab_size+1, 
                                                config.hidden_size//config.n_concat_token)
            
        if config.use_init_prob:
             self.use_init_embeddings = nn.Embedding(2, config.hidden_size)

        self.layers = nn.ModuleList()
        dpr = config.use_drop_path if config.use_drop_path else 0
        dpr = [x.item() for x in torch.linspace(0, dpr, config.num_hidden_layers)]
        for i in range(config.num_hidden_layers):
            self.layers.append(Block(config.hidden_size, config.num_attention_heads, config, drop_path=dpr[i]))

        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size//config.n_concat_token, config.vocab_size, bias=False)
        if config.classify:
            self.clf_head = nn.Linear(config.hidden_size, config.num_classes)
            
        self.n_concat_token = config.n_concat_token
        self.add_pos_first = config.add_pos_first
        self.double_pos = config.double_pos
        self.well_cond_prob = config.well_cond_prob
        self.use_dip = config.use_dip
        self.dip_bins = config.dip_bins
        self.vqvae_refl_dir = config.vqvae_refl_dir
        self.refl_vocab_size = config.refl_vocab_size
        self.position_embedding_type = config.position_embedding_type
        self.add_dip_to_well = config.add_dip_to_well
        self.prepend_refl = config.prepend_refl
        self.vocab_size = config.vocab_size
        self.use_init_prob = config.use_init_prob
        self.broadcast_glob_pos = config.broadcast_glob_pos
        self.adaln_glob_pos = config.adaln_glob_pos
    
    def forward(self, x, cls=None, well_pos=None, well_token=None, dip=None, refl=None, dip_well=None, init=None, input_pos=None, attn_mask=None):
        length, batch = x.shape
        if getattr(self.layers[0].attn, 'kv_cache', None) is not None:
            cache_is_none = torch.all(self.layers[0].attn.kv_cache.k_cache == 0)
        else:
            cache_is_none = True
        cache_is_none = cache_is_none and getattr(self.layers[0], 'linear_state', None) is None
        
        h = self.token_embeddings(x)
        
        if self.add_pos_first and self.position_embedding_type == "learnable":
            positions = torch.arange(length, device=x.device).unsqueeze(-1)
            h = h + self.position_embeddings(positions).expand_as(h)
        
        if self.n_concat_token > 1:
#             h = h.transpose(1, 2)
#             h = h.reshape(length//self.n_concat_token, -1, batch)
#             h = h.transpose(1, 2)
            h = h.reshape(length//self.n_concat_token, batch, -1)
        
        # prepend sos/cls token
        if not self.cls_token:
            sos = torch.ones(1, batch, self.hidden_size, device=x.device) * self.sos
            h = torch.cat([sos, h[:-1, :, :]], axis=0) if len(h) > 2 or cache_is_none else h # For KV caching support
        elif self.cls_token and cls is not None:
            cls = self.cls_token_embeddings(cls).unsqueeze(0)
            h = torch.cat([cls, h[:-1, :, :]], axis=0) if len(h) > 2 or cache_is_none else h # For KV caching support
            
        if self.use_dip and dip is not None:
            use_dip_embed = torch.ones(dip.shape[0], device=h.device).long()
            use_dip_embed[dip[:, 0] == len(self.dip_bins)] = 0
            use_dip_embed = self.use_dip_embeddings(use_dip_embed).unsqueeze(0)
            dip_embed = self.dip_embeddings(dip).transpose(0, 1)
            h += dip_embed
            h = torch.cat([use_dip_embed, h], axis=0) if len(h) > 2 or cache_is_none else h # For KV caching support
            
        if self.vqvae_refl_dir is not None and refl is not None:
            use_refl_embed = torch.ones(refl.shape[1], device=h.device).long()
            use_refl_embed[refl[0, :] == self.refl_vocab_size] = 0
            use_refl_embed = self.use_refl_embeddings(use_refl_embed).unsqueeze(0)
            refl_embed = self.refl_embeddings(refl)
            if self.use_dip and dip is not None and self.broadcast_glob_pos:
                refl_embed += dip_embed
            if self.prepend_refl:
                h = torch.cat([use_refl_embed, refl_embed, h], axis=0)
            else:
                h[1+self.use_dip:] += refl_embed[1:]
                h = torch.cat([use_refl_embed, h], axis=0)

        if self.well_cond_prob > 0 and well_pos is not None and well_token is not None:
            well_pos_embed = self.well_position_embeddings(well_pos).unsqueeze(0)
            well_embed = self.token_embeddings(well_token)
            if self.use_dip and self.add_dip_to_well and dip_well is not None:
                dip_well_embed = self.dip_embeddings(dip_well).transpose(0, 1)
                well_embed += dip_well_embed
            h = torch.cat([well_pos_embed, well_embed, h], axis=0)

        if self.use_init_prob and init is not None:
            use_init_embed = torch.ones(init.shape[1], device=h.device).long()
            use_init_embed[init[0, :] == self.vocab_size] = 0
            use_init_embed = self.use_init_embeddings(use_init_embed).unsqueeze(0)
            init_embed = self.token_embeddings(init)
            if self.use_dip and dip is not None and self.broadcast_glob_pos:
                init_embed += dip_embed
            h = torch.cat([use_init_embed, init_embed, h], axis=0)
            
        if not self.add_pos_first and self.position_embedding_type == "learnable":
            # add positional embeddings
            if self.well_cond_prob > 0 or self.use_dip or self.vqvae_refl_dir is not None:
                positions = torch.arange(len(h)//self.n_concat_token, device=x.device).unsqueeze(-1) if input_pos is None else input_pos
            else:
                positions = torch.arange(length//self.n_concat_token, device=x.device).unsqueeze(-1) if input_pos is None else input_pos
            h = h + self.position_embeddings(positions).expand_as(h)
            
        if self.double_pos and not self.position_embedding_type == "learnable":
            positions = torch.arange(length//self.n_concat_token, device=x.device).unsqueeze(-1) if input_pos is None else input_pos
            h = h + self.position_embeddings2(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h, pos=dip_embed if self.adaln_glob_pos else None, attn_mask=attn_mask)

        h = self.ln_f(h)
        
        if self.n_concat_token > 1:
#             h = h.transpose(1, 2)
#             h = h.reshape(length, -1, batch)
#             h = h.transpose(1, 2)
            h = h.reshape(length, batch, -1)
            
        logits = self.head(h)
        
        if self.use_dip and dip is not None:
            logits = logits[1:] if len(logits) >= 2 else logits # For KV caching support
            
        if self.vqvae_refl_dir is not None and refl is not None:
            logits = logits[1:]
            if self.prepend_refl:
                logits = logits[len(refl_embed):]
        
        if self.well_cond_prob > 0 and well_pos is not None and well_token is not None:
            logits = logits[len(well_embed)+1:]
        
        if self.use_init_prob and init is not None:
            logits = logits[len(init_embed)+1:]
                
        if not self.classify:
            # return logits
            return logits
        
        h = torch.mean(h, dim=0)  # average pool over sequence
        # return classification logits and generative logits
        return self.clf_head(h), logits

class UNet2(nn.Module):
    """UNet architecture
    UNet architecture composed of a series of contracting blocks followed by expanding blocks.
    Most UNet implementations available online hard-code a certain number of levels. Here,
    the number of levels for the contracting and expanding paths can be defined by the user and the
    UNet is built in such a way that the same code can be used for any number of levels without modification.
    """

    def __init__(self, input_channels=1, output_channels=1, hidden_channels=64, levels=2, kernel_size=3, use_dropout=False, dropout_prob=0.1, use_bn=False, config=None):
        super(UNet2, self).__init__()
        self.levels = levels
        self.upfeature = FeatureMapBlock2(input_channels, hidden_channels)
        self.contracts = nn.ModuleList([ContractingBlock2(hidden_channels * (2 ** level), kernel_size=kernel_size, use_dropout=use_dropout, dropout_prob=dropout_prob, use_bn=use_bn) for level in range(levels)])
        self.bottleneck = Bottle_neck2(hidden_channels * (2 ** levels), kernel_size=kernel_size, use_bn=use_bn, use_dropout=use_dropout, dropout_prob=dropout_prob)
        self.expands = nn.ModuleList([ExpandingBlock2(hidden_channels * (2 ** (levels - level+1)), kernel_size=kernel_size, use_bn=use_bn, use_dropout=use_dropout, dropout_prob=dropout_prob) for level in range(levels)])
        self.downfeature1 = FeatureMapBlock2(hidden_channels*(2**1), hidden_channels)
        # self.downfeature2 = FeatureMapBlock(hidden_channels*(2**1), output_channels)
        self.downfeature2 = FeatureMapBlock2(hidden_channels, output_channels)

        self.mask_token = nn.Parameter(torch.empty(1, 1, config.vocab_size), requires_grad=True)
        nn.init.uniform_(self.mask_token, -1, 1)

    def forward(self, x, **kwargs):
        xenc = []
        x = self.upfeature(x)
        # xenc.append(x)
        for level, contract in enumerate(self.contracts):
            x1,x = contract(x)
            xenc.append(x1)
        # print('len(xenc):', len(xenc))
        # print('xenc[-1].shape:', xenc[-1].shape)
        # print('xenc[0].shape:', xenc[0].shape)
        x = self.bottleneck(x)
        for level, expand in enumerate(self.expands):
            x = expand(x, xenc[self.levels - level - 1])
        x = self.downfeature1(x)
        x = self.downfeature2(x).squeeze(1)
        return x