import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import lightning.pytorch as pl
from .pytorch_modules.layers import *
from .pytorch_modules.functional import masked_mean, convert_attn_mask_mha

# Paper: https://www.honda-ri.de/pubs/pdf/4503.pdf

### Model for final prediction (Does not have training layers)
class EncoderBackbone(pl.LightningModule):
    def __init__(self, num_df, d_model, heads=6, num_layers=4, features=32, scaler=2, dropout=0.1, 
                 embedding_version='v1', emb_k=16, emb_p=2., tf_pre_norm=True, stride=1):
        super().__init__()
        assert isinstance(stride, int), f'stride must be of type int but is {type(stride)}!'
        self.num_df, self.d_model, self.emb_k = num_df, d_model, emb_k
        self.features, self.stride, self.heads = features, stride, heads
        
        if embedding_version == 'v1':
            self.emb_layer = KNNEmbedding(num_df, d_model, emb_k, p=emb_p, dropout=dropout)
        elif embedding_version == 'v2':
            self.emb_layer = KNNEmbeddingV2(d_model, emb_k, p=emb_p, dropout=dropout)
        elif embedding_version == 'v3':
            self.emb_layer = KNNEmbeddingV3(num_df, d_model, emb_k, p=emb_p, dropout=dropout)
        else:
            raise 'Embedding Layer version not recognized!'
        
        self.out_norm = nn.LayerNorm(d_model)
        self.feature_extractor = FeatureExtractor(d_model, features, dropout=dropout)
        
        self.layers = []
        for _ in range(num_layers):
            self.layers += [TransformerLayer(d_model, heads, outdim=None, ff_scaler=scaler, dropout=dropout, 
                                             pre_norm=tf_pre_norm)]
        self.layers = nn.ModuleList(self.layers)
        self.tracked_attn = []
        self.save_hyperparameters()
            
    def forward(self, x, features, attn_mask=None, stride=None, track_attn=False):
        assert len(x.shape) == 3, f'Shape of x must be of length three: Batch, Num, Dim; but is {x.shape}!'
        assert attn_mask is None or len(attn_mask.shape) == 3, f'Shape of attn_mask must be None or of length three: ' + \
                                                               f'Batch, Num, 1; but is {attn_mask.shape}!'
        assert features.shape == (x.shape[0], x.shape[2]), f'Features ({features.shape}) must have same dimensions \
                                                            as x ({x.shape}) except for secodn dim!'
        assert isinstance(stride, int) or stride is None, f'stride must be of type int or None but is {type(stride)}!'
        stride = self.stride if stride is None else stride
        
        ## Execute encoding and embedding.
        n_batch, n_num, n_dim = x.shape
        x, _ = self.emb_layer(x, features, attn_mask) # Execute Embedding
        
        ## Execute stride
        if stride > 1:
            x = x[:,::stride,:]
            
        if self.stride > 1 and attn_mask is not None:
            attn_mask = attn_mask[:,::stride,:]
        
        ## Execute Transformer Encoder
        attn_mask_mha = convert_attn_mask_mha(attn_mask, self.heads) if attn_mask is not None else None
        for i,layer in enumerate(self.layers):
            x, attn = layer(x, attn_mask_mha, return_attention=track_attn)
            self.tracked_attn += [attn.detach().cpu()] if track_attn else [] # Save attn if required
        
        ## Execute last layer norm and feature extractor
        x = self.out_norm(x)
        x_feat = self.feature_extractor(x, attn_mask)
        return x_feat, x, attn_mask
    
    def predict(self, coordinates, fvalues, **kwargs):
        n_points, n_dim = coordinates.shape
        fvalues = np.expand_dims(fvalues, 1) if len(fvalues.shape) == 1 else fvalues
            
        ## Prepare data
        data = np.concatenate([coordinates, fvalues], axis=1)
        n_dim_total = data.shape[1]
        x = np.zeros((n_points, self.num_df))
        features = np.zeros(self.num_df)
        features[n_dim:] = 1
        x[:,:n_dim_total] = data
        
        ## Convert to tensor
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        ## Apply model
        x_feat, emb = self.predict_batch(x=x, features=features, **kwargs)
        if emb is not None:
            return x_feat.cpu().squeeze(0).numpy(), emb.cpu().squeeze(0).numpy()
        else:
            return x_feat.cpu().squeeze(0).numpy()
        
    def predict_batch(self, *args, return_embeddings=False, **kwargs):
        with torch.no_grad():
            x_feat, x, _ = self.forward(*args, **kwargs)
        return (x_feat, x) if return_embeddings else (x_feat, None)
    
        
    def training_step(self, batch, batch_idx):
        assert False, 'Don\'t use this model for training. Instead use \'EncoderTraining\'!'
        pass
    
    def configure_optimizers(self):
        assert False, 'Don\'t use this model for training. Instead use \'EncoderTraining\'!'
        return None