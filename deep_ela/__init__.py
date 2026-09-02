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
## `stride` defaults to the value each checkpoint was published with: 1 for the medium
## models, 2 for the large ones. `stride=1` is RECOMMENDED for the large models -- it is
## what forward() uses to keep every stride-th token after the kNN embedding, so stride 2
## sees half the points. Measured on BBOB at dim 5, stride 2 costs about twice the feature
## noise and about twice the sample budget, for roughly 2.5x the throughput. Pass
## `stride=None` to follow the checkpoint explicitly.
##
## `repetitions` averages predict() over random permutations of the num_df slots. Feature
## quality saturates by about 10; beyond that it only buys reproducibility (the jitter falls
## as 1/sqrt(repetitions)), so use 50 or more when a number has to reproduce.
def load_medium_25d_v1(device='cpu', stride=1, repetitions=10):
    name = 'deep_ela.medium_25d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/medium_25d_v1', device=device,
                   stride=stride, repetitions=repetitions).train(False)

def load_medium_50d_v1(device='cpu', stride=1, repetitions=10):
    name = 'deep_ela.medium_50d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/medium_50d_v1', device=device,
                   stride=stride, repetitions=repetitions).train(False)

## stride=2 is what these two shipped with; stride=1 gives markedly better features.
def load_large_25d_v1(device='cpu', stride=2, repetitions=10):
    name = 'deep_ela.large_25d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/large_25d_v1', device=device,
                   stride=stride, repetitions=repetitions).train(False)

def load_large_50d_v1(device='cpu', stride=2, repetitions=10):
    name = 'deep_ela.large_50d_v1'
    return DeepELA(name, path_dir=f'{BASE_DIR}/models/large_50d_v1', device=device,
                   stride=stride, repetitions=repetitions).train(False)
