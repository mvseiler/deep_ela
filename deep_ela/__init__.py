from .encoders import EncoderBackbone
from glob import glob
import os, yaml, torch, time

BASE_DIR = os.path.dirname(__file__)

## Wrapper class
class DeepELA(EncoderBackbone):
    def __init__(self, name, path_dir=None, path_ckpt=None, path_cnfg=None, device='cpu'):
        self.name = name
        assert path_dir is not None or (path_ckpt is not None and path_cnfg is not None), \
            'Either path_dir or path_ckpt and path_config must be provided!'
        ## Identify checkpoint and config paths
        if path_dir is not None:
            try:
                path_ckpt = glob(os.path.join(path_dir, '*_backbone.ckpt'))[0]
                path_cnfg = glob(os.path.join(path_dir, 'hparams.yaml'))[0]
            except:
                raise Exception(f'Failed to load model at location {path_dir}!')
        ## Load config and create model
        with open(path_cnfg) as f:
            config = yaml.safe_load(f.read())
        super().__init__(**config)
        ## Load parameters
        self.load_state_dict(torch.load(path_ckpt, weights_only=True)) 
        ## Set device and eval
        self.to(device).eval()
        
    def __call__(self, X, y, include_costs=False, repetitions=10):
        start = time.time() # Measure runtime
        features = super().predict(coordinates=X, fvalues=y, repetitions=repetitions, return_embeddings=False)
        features = {f'{self.name}.X{i}': f for i,f in enumerate(features)}
        if include_costs:
            features[f'{self.name}.costs_runtime'] = time.time() - start
        return features
    

## Simple loader functions    
def load_medium_25d_v1(device='cpu'):
    name = 'deep_ela.medium_25d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/medium_25d_v1', device=device).train(False)

def load_medium_50d_v1(device='cpu'):
    name = 'deep_ela.medium_50d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/medium_50d_v1', device=device).train(False)
    
def load_large_25d_v1(device='cpu'):
    name = 'deep_ela.large_25d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/large_25d_v1', device=device).train(False)

def load_large_50d_v1(device='cpu'):
    name = 'deep_ela.large_50d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/large_50d_v1', device=device).train(False)