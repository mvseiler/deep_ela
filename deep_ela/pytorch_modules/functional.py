import torch, torch.nn as nn, torch.nn.functional as F
from torch.masked import masked_tensor
from einops import repeat
import math

## Find kNNs
def find_knn(x, dist, k, attn_mask=None, project_local=True, flatten=True):
    if attn_mask is not None: # Make unallowed distanzes very large
        dist_mask = (attn_mask.unsqueeze(1) | attn_mask.unsqueeze(2)).squeeze(-1)
        dist = torch.where(dist_mask, 1e6, dist)
    _, idx = dist.topk(k=k, dim=-1, largest=False) # Select kNN
    
    idx = repeat(idx, 'b n k -> b n k f', f=x.shape[-1]) # Repeat that idx matches X
    x = x.unsqueeze(2).expand_as(idx)
    x = x.gather(1, idx) # x.shape (B,N,K,F)
    if project_local: # Project to local neighborhood and restandardize
        x[:,:,1:,:] = (x[:,:,1:,:] - x[:,:,:1,:]) / math.sqrt(2.) 
    x = x.flatten(2) if flatten else x.transpose(-2, -1)
    return x, idx[...,0]

## Following function ensures that attn_mask is bool and not float or int
## True or values > 0. indicates position to be ignored. 
def convert_attn_mask(attn_mask):
    if attn_mask.dtype != torch.bool:
        return attn_mask > (0. + 1e-5)
    else:
        return attn_mask 

## Sums up x along dim and takes attn_mask into account if provided
def masked_sum(x, attn_mask=None, dim=1, keepdim=False):
    if attn_mask is None:
        return x.sum(dim, keepdim=keepdim)
    
    attn_mask_ = convert_attn_mask(attn_mask)
    return x.masked_fill(attn_mask_, 0.).sum(dim, keepdim)

## Calc. mean of x along dim and takes attn_mask into account if provided
def masked_mean(x, attn_mask=None, dim=1, keepdim=False):
    if attn_mask is None:
        return x.mean(dim, keepdim=keepdim)
    
    attn_mask_ = convert_attn_mask(attn_mask)
    x = x.masked_fill(attn_mask_, 0.).sum(dim, keepdim)
    denominator = (~attn_mask_).sum(dim, keepdim) + 1e-5
    return x / denominator

## Calc. var of x along dim and takes attn_mask into account if provided
def masked_var(x, attn_mask=None, dim=1, keepdim=False):
    if attn_mask is None:
        return x.var(dim, keepdim=keepdim)
    
    x_mean = masked_mean(x, attn_mask, dim=dim, keepdim=True)
    diff = (x - x_mean).pow(2.)
    return masked_mean(diff, attn_mask, dim=dim, keepdim=keepdim)

## Calc. std of x along dim and takes attn_mask into account if provided
def masked_std(x, attn_mask=None, dim=1, keepdim=False):
    return (masked_var(x, attn_mask=attn_mask, dim=dim, keepdim=keepdim) + 1e-5).sqrt()

## Detaches gradient of x at position attn_mask without filling values to zero. Might be helpfull when updating running mean, std
def masked_fill(x, attn_mask=None):
    if attn_mask is None:
        return x
    
    attn_mask_ = convert_attn_mask(attn_mask)
    x_detach = x.detach()
    return torch.where(attn_mask_, x_detach, x)

## Converts attn_mask given to attn_mask as needed by MHA Layer!
def convert_attn_mask_mha(attn_mask, num_heads):
    n_batch, n_seq = attn_mask.shape[:2]
    # get correct Attention Mask (twice for both directions)
    attn_mask_mha  = repeat(attn_mask.squeeze(-1), 'b n -> (b h) n m', m=n_seq, h=num_heads)
    attn_mask_mha |= repeat(attn_mask.squeeze(-1), 'b n -> (b h) m n', m=n_seq, h=num_heads)
    return attn_mask_mha