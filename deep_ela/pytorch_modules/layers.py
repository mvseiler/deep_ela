import math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.distributed as dist
from einops import repeat
from .functional import masked_mean, masked_std, find_knn, convert_attn_mask_mha

class FeatureExtractor(nn.Module):
    def __init__(self, indim, outdim, use_glu=False, bias=False, dropout=0.1):
        super().__init__()
        outdim = 2 * outdim if use_glu else outdim
        self.layer  = [nn.Linear(indim, outdim, bias=bias)]
        if use_glu:
            self.layer += [nn.GLU()]
        self.layer += [nn.Dropout(dropout)]
        self.layer = nn.Sequential(*self.layer)
    
    def forward(self, x, attn_mask=None):
        x_out = self.layer(x)
        # Ignore pooling if we have only (B,F). That is the case if we use the feature token
        if len(x.shape) >= 3:
            x_out = masked_mean(x_out, attn_mask)
        return x_out.tanh()


class FeedFoward(nn.Module):
    def __init__(self, indim, hiddim, outdim=None, use_glu=False, bias=False, dropout=0.1, final_af=False, final_dp=False):
        super().__init__()
        if not isinstance(hiddim, (list, tuple)):
            hiddim = [hiddim]
        outdim = indim if outdim is None else outdim
            
        activation, scaler = (nn.GLU, 2) if use_glu else (nn.GELU, 1)
        layers = []
        inp = [indim] + hiddim[:-1]
        out = hiddim
        outdim = 2*outdim if final_af and use_glu else outdim
        
        for i,o in zip(inp, out):
            layers.append(nn.Linear(i, scaler*o, bias=bias))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(out[-1], outdim, bias=bias))
        
        if final_af:
            layers.append(activation())
        
        if final_dp:
            layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class KNNEmbedding(nn.Module):
    def __init__(self, num_df, d_model, k, p=2.0, scaler=2, use_glu=True, dropout=0.1):
        super().__init__()
        self.k, self.p, self.num_df = int(k), p, int(num_df)
        self.ff = nn.Sequential(
            nn.Linear(num_df*k*2, 2*d_model),
            nn.GLU(),
            nn.Dropout(dropout),
        )
    
    def _forward(self, x, features, attn_mask=None, masking=0.):
        n_batch, n_points, n_dim = x.shape
        n_target = 2 * n_dim
        features_ = features.unsqueeze(1) > 0.1
        
        ## Standardize x but save unstandardized coord values for kNN
        x_crd = x.masked_fill(features_, 0.)
        x_ftr = x.masked_fill(~features_, 0.)
        x = torch.cat([x_crd, x_ftr], dim=-1)
        x_mean, x_std = masked_mean(x, attn_mask, 1, keepdim=True), masked_std(x, attn_mask, 1, keepdim=True)
        x = (x - x_mean) / (x_std + 1e-5)
        x = x.clamp(min=-10, max=10) # Clamp very large values
        
        # Find kNN if k is larger 1 (First point is its own neighbor)
        if self.k >= 2:
            dist = torch.cdist(x_crd, x_crd, p=self.p)
            x, idx = find_knn(x, dist, self.k, attn_mask=attn_mask, project_local=True)
                    
        # Apply point masking (needed for training)
        if 0. < masking <= 1.:
            x_gt = x[...,:n_target].clone() # (B,N,2*F)
            point_mask = torch.rand(n_batch, n_points, 1).to(x)
            point_mask = point_mask < masking # This is where we mask
            if attn_mask is not None:
                point_mask = point_mask & ~attn_mask # We don't want to train on masked points!
            randn_values = torch.randn_like(x[...,:n_target])
            x[...,:n_target] = torch.where(point_mask, randn_values, x[...,:n_target]) # Fill random values
            return x, {'x_gt': x_gt.detach(), 'point_mask': point_mask, 'attn_mask': attn_mask}
        else:
            return x, None
    
    def forward(self, x, features, attn_mask=None, masking=0.):
        x, pm_info = self._forward(x, features, attn_mask=attn_mask, masking=masking)
        x = self.ff(x)
        return x, pm_info
    
class TransformerLayer(nn.MultiheadAttention):
    def __init__(self, d_model, heads, outdim=None, ff_scaler=2, dropout=0.1, 
                 gelu=False, ff_glu=True, beta_init=1e-5):
        super().__init__(d_model, heads, dropout=dropout, bias=False, batch_first=True, add_zero_attn=True)
        
        self.out = nn.Linear(d_model, outdim, bias=False) if outdim is not None else None
        self.gelu, self.heads = gelu, heads
        self.ff = FeedFoward(d_model, ff_scaler*d_model, outdim=outdim, use_glu=ff_glu, bias=False, 
                             dropout=dropout, final_dp=True)
                
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        
    def forward(self, x, attn_mask=None, return_attention=False):
        ## Create attention mask
        if attn_mask is not None and len(attn_mask.shape) == 2:
            assert attn_mask.shape[:2] == x.shape[:2], \
                    f'Attn Mask with shape {attn_mask.shape} must match shape of X {x.shape} other than last dim.'
            attn_mask_mha = convert_attn_mask_mha(attn_mask, self.num_heads)
        elif attn_mask is not None and len(attn_mask.shape) == 3:
            shape = (self.num_heads * x.shape[0], x.shape[1], x.shape[1])
            assert attn_mask.shape == shape, \
                    f'Attn Mask with shape {attn_mask.shape} must match shape of {shape}.'
            attn_mask_mha = attn_mask
        else:
            attn_mask_mha = None
        
        ## MHA
        x_norm = self.norm1(x)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            x_out, attn = super().forward(x_norm, x_norm, x_norm, attn_mask=attn_mask, 
                                          need_weights=return_attention, average_attn_weights=True)
        x = F.gelu(x + x_out) if self.gelu else (x + x_out)
        
        ## FF
        x_norm = self.norm2(x)
        x = self.out(x) if self.out is not None else x
        x_out = self.ff(x_norm)
        x = F.gelu(x + x_out) if self.gelu else (x + x_out)
        
        ## Return
        if return_attention:
            return x, attn
        else:
            return x, None