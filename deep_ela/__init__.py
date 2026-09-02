from .encoders import EncoderBackbone
from glob import glob
import os, yaml, torch, time

BASE_DIR = os.path.dirname(__file__)

## Wrapper class
class DeepELA(EncoderBackbone):
    def __init__(self, name, path_dir=None, path_ckpt=None, path_cnfg=None, device='cpu',
                 stride=None, repetitions=10):
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
        ## Token stride. forward() keeps every stride-th token after the kNN embedding, so a
        ## stride-2 model attends over half its points. The published checkpoints disagree --
        ## medium ships 1, large ships 2 -- and stride 2 costs roughly a factor of two in both
        ## feature noise and required sample size for about 2.5x the throughput. Overriding the
        ## config here rather than per call keeps self.stride and the call-time stride in
        ## agreement, which is also what stops forward()'s two stride guards from diverging.
        if stride is not None:
            assert isinstance(stride, int) and stride >= 1, \
                f'stride must be a positive int but is {stride!r}!'
            config['stride'] = stride
        self.repetitions = repetitions
        super().__init__(**config)
        ## Load parameters
        self.load_state_dict(torch.load(path_ckpt, weights_only=True)) 
        ## Set device and eval
        self.to(device).eval()
        
    def __call__(self, X, y, include_costs=False, repetitions=None, stride=None):
        """`repetitions` and `stride` both default to the values this model was
        constructed with; pass either to override it for one call, which is the
        cheap way to see what the setting costs on your own problems. `stride`
        is handed straight to forward(), which falls back to self.stride when it
        is None, so there is nothing to resolve here."""
        start = time.time() # Measure runtime
        repetitions = self.repetitions if repetitions is None else repetitions
        features = super().predict(coordinates=X, fvalues=y, repetitions=repetitions,
                                   stride=stride, return_embeddings=False)
        features = {f'{self.name}.X{i}': f for i,f in enumerate(features)}
        if include_costs:
            features[f'{self.name}.costs_runtime'] = time.time() - start
        return features
    

## Simple loader functions
##
## `stride` and `repetitions` may be set here or overridden per call. `stride=None` takes
## whatever the checkpoint's hparams.yaml specifies, which is 1 for the medium models and 2
## for the large ones.
def load_medium_25d_v1(device='cpu', stride=None, repetitions=10):
    name = 'deep_ela.medium_25d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/medium_25d_v1', device=device,
                   stride=stride, repetitions=repetitions).train(False)

def load_medium_50d_v1(device='cpu', stride=None, repetitions=10):
    name = 'deep_ela.medium_50d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/medium_50d_v1', device=device,
                   stride=stride, repetitions=repetitions).train(False)

def load_large_25d_v1(device='cpu', stride=None, repetitions=10):
    name = 'deep_ela.large_25d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/large_25d_v1', device=device,
                   stride=stride, repetitions=repetitions).train(False)

def load_large_50d_v1(device='cpu', stride=None, repetitions=10):
    name = 'deep_ela.large_50d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/large_50d_v1', device=device,
                   stride=stride, repetitions=repetitions).train(False)
