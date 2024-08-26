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

        self.token_embeddings = nn.Embedding(config.vocab_size+math.ceil(config.well_cond_prob), 
                                             config.hidden_size//config.n_concat_token)
        
        if config.position_embedding_type != "alibi":
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

        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(Block(config.hidden_size, config.num_attention_heads, config))

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
    
    def forward(self, x, cls=None, well_pos=None, well_token=None, dip=None, refl=None, dip_well=None):
        length, batch = x.shape
        
        h = self.token_embeddings(x)
        
        if self.add_pos_first and self.position_embedding_type != "alibi":
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
            h = torch.cat([sos, h[:-1, :, :]], axis=0)
        elif self.cls_token and cls is not None:
            cls = self.cls_token_embeddings(cls).unsqueeze(0)
            h = torch.cat([cls, h[:-1, :, :]], axis=0)
            
        if self.use_dip and dip is not None:
            use_dip_embed = torch.ones(dip.shape[0], device=h.device).long()
            use_dip_embed[dip[:, 0] == len(self.dip_bins)] = 0
            use_dip_embed = self.use_dip_embeddings(use_dip_embed).unsqueeze(0)
            dip_embed = self.dip_embeddings(dip).transpose(0, 1)
            h[1:] += dip_embed[1:]
            h = torch.cat([use_dip_embed, h], axis=0)
            
        if self.vqvae_refl_dir is not None and refl is not None:
            use_refl_embed = torch.ones(refl.shape[1], device=h.device).long()
            use_refl_embed[refl[0, :] == self.refl_vocab_size] = 0
            use_refl_embed = self.use_refl_embeddings(use_refl_embed).unsqueeze(0)
            refl_embed = self.refl_embeddings(refl)
            h[1:] += refl_embed[1:]
            h = torch.cat([use_refl_embed, h], axis=0)
        
        if self.well_cond_prob > 0 and well_pos is not None and well_token is not None:
            well_pos_embed = self.well_position_embeddings(well_pos).unsqueeze(0)
            well_embed = self.token_embeddings(well_token)
            if self.use_dip and self.add_dip_to_well and dip_well is not None:
                dip_well_embed = self.dip_embeddings(dip_well).transpose(0, 1)
                well_embed += dip_well_embed
            h = torch.cat([well_pos_embed, well_embed, h], axis=0)
            
        if not self.add_pos_first and self.position_embedding_type != "alibi":
            # add positional embeddings
            if self.well_cond_prob > 0 or self.use_dip or self.vqvae_refl_dir is not None:
                positions = torch.arange(len(h)//self.n_concat_token, device=x.device).unsqueeze(-1)
            else:
                positions = torch.arange(length//self.n_concat_token, device=x.device).unsqueeze(-1)
            h = h + self.position_embeddings(positions).expand_as(h)
            
        if self.double_pos and not self.position_embedding_type != "alibi":
            positions = torch.arange(length//self.n_concat_token, device=x.device).unsqueeze(-1)
            h = h + self.position_embeddings2(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        
        if self.n_concat_token > 1:
#             h = h.transpose(1, 2)
#             h = h.reshape(length, -1, batch)
#             h = h.transpose(1, 2)
            h = h.reshape(length, batch, -1)
            
        logits = self.head(h)
        
        if self.use_dip and dip is not None:
            logits = logits[1:]
            
        if self.vqvae_refl_dir is not None and refl is not None:
            logits = logits[1:]
        
        if self.well_cond_prob > 0 and well_pos is not None and well_token is not None:
            logits = logits[len(well_embed)+1:]
                
        if not self.classify:
            # return logits
            return logits
        
        h = torch.mean(h, dim=0)  # average pool over sequence
        # return classification logits and generative logits
        return self.clf_head(h), logits