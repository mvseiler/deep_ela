import math
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from einops import repeat
from .functional import masked_mean, masked_std, find_knn, convert_attn_mask_mha

class FeatureExtractor(nn.Module):
    def __init__(self, indim, outdim, dropout=0.1):
        super().__init__()
        self.layer  = nn.Sequential(
            nn.Linear(indim, 2 * outdim, bias=False),
            nn.GLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x, attn_mask=None):
        x_out = self.layer(x)
        # Ignore pooling if we have only (B,F). That is the case if we use the feature token
        if len(x.shape) >= 3:
            x_out = masked_mean(x_out, attn_mask)
        return x_out.tanh()


class FeedFoward(nn.Module):
    def __init__(self, indim, hiddim, outdim=None, dropout=0.1, final_af=False, final_dp=False):
        super().__init__()
        if not isinstance(hiddim, (list, tuple)):
            hiddim = [hiddim]
        outdim = indim if outdim is None else outdim
            
        layers = []
        inp = [indim] + hiddim[:-1]
        out = hiddim
        outdim = 2*outdim if final_af else outdim
        
        for i,o in zip(inp, out):
            layers.append(nn.Linear(i, 2*o, bias=False))
            layers.append(nn.GLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(out[-1], outdim, bias=False))
        
        if final_af:
            layers.append(activation())
        
        if final_dp:
            layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class KNNEmbedding(nn.Module):
    def __init__(self, num_df, d_model, k, p=2.0, dropout=0.1):
        super().__init__()
        self.k, self.p, self.num_df = int(k), p, int(num_df)
        self.ff = nn.Sequential(
            nn.Linear(num_df*k*2, 2*d_model),
            nn.GLU(),
            nn.Dropout(dropout),
        )
    
    def _forward(self, x, features, attn_mask=None):
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
            x, _ = find_knn(x, dist, self.k, attn_mask=attn_mask, project_local=True)
        return x
    
    def forward(self, x, features, attn_mask=None):
        x = self._forward(x, features, attn_mask=attn_mask)
        x = self.ff(x)
        return x, None
    
    
class KNNEmbeddingV2(nn.Module):
    def __init__(self, d_model, k, p=2.0, dropout=0.1):
        super().__init__()
        self.k, self.p = int(k), p
        self.ff_crd = nn.Linear(k, 4*d_model, bias=False)
        self.ff_ftr = nn.Linear(k, 4*d_model, bias=False)
        self.out = nn.Sequential(
            nn.GLU(),
            nn.Dropout(dropout),
            nn.Linear(2*d_model, d_model, bias=False),
        )
    
    def _forward(self, x, features, attn_mask=None):
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
            x, _ = find_knn(x, dist, self.k, attn_mask=attn_mask, project_local=True, flatten=False)
        return x[:,:,:n_dim,:], x[:,:,n_dim:,:] # Split into Coords and Features
    
    def forward(self, x, features, attn_mask=None):
        x_crd, x_ftr = self._forward(x, features, attn_mask=attn_mask)
        x = self.ff_crd(x_crd).sum(2) + self.ff_ftr(x_ftr).sum(2) # Execute forward projection, take sum
        return self.out(x), None
    
    
class KNNEmbeddingV3(nn.Module):
    def __init__(self, num_df, d_model, k, p=2.0, dropout=0.1):
        super().__init__()
        self.k, self.p = int(k), p
        self.ff_crd = nn.Linear(k, d_model, bias=False)
        self.ff_ftr = nn.Linear(k, d_model, bias=False)
        
        self.pe_crd = nn.Parameter(0.01*torch.randn(1, 1, num_df, d_model))
        self.pe_ftr = nn.Parameter(0.01*torch.randn(1, 1, num_df, d_model))
        
        self.dropout = nn.Dropout(dropout)
       
    
    def _forward(self, x, features, attn_mask=None):
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
            x, _ = find_knn(x, dist, self.k, attn_mask=attn_mask, project_local=True, flatten=False)
        return x[:,:,:n_dim,:], x[:,:,n_dim:,:] # Split into Coords and Features
    
    def forward(self, x, features, attn_mask=None, masking=0.):
        x_crd, x_ftr = self._forward(x, features, attn_mask=attn_mask)
        x_crd = (self.ff_crd(x_crd) + self.pe_crd).sum(2)
        x_ftr = (self.ff_ftr(x_ftr) + self.pe_ftr).sum(2)
        pe_loss = self.pe_crd.abs().sum() + self.pe_ftr.abs().sum()
        return self.dropout(x_crd + x_ftr), pe_loss
    
    
class TransformerLayer(nn.MultiheadAttention):
    def __init__(self, d_model, heads, outdim=None, ff_scaler=2, dropout=0.1, pre_norm=True):
        super().__init__(d_model, heads, dropout=dropout, bias=False, batch_first=True, add_zero_attn=True)
        
        self.out = nn.Linear(d_model, outdim, bias=False) if outdim is not None else None
        self.pre_norm, self.heads = pre_norm, heads
        self.ff = FeedFoward(d_model, ff_scaler*d_model, outdim=outdim, dropout=dropout, final_dp=True)
                
        if pre_norm:
            self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        else:
            self.norm1, self.norm2 = nn.Identity(), nn.Identity()
        
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
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True):
            x_out, attn = super().forward(x_norm, x_norm, x_norm, attn_mask=attn_mask, 
                                          need_weights=return_attention, average_attn_weights=True)
        x = (x + x_out)
        
        ## FF
        x_norm = self.norm2(x)
        x = self.out(x) if self.out is not None else x
        x_out = self.ff(x_norm)
        x = (x + x_out)
        
        ## Return
        if return_attention:
            return x, attn
        else:
            return x, None