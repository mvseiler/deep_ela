# Deep ELA: Deep Exploratory Landscape Analysis with Self-Supervised Pretrained Transformers for Single- and Multi-Objective Continuous Optimization Problems
This package is an extention to the pflacco package: [https://github.com/Reiyan/pflacco.git](https://github.com/Reiyan/pflacco.git). Deep-ELA is still in an early development phase. Thus be cautious!

## Setup
To include Deep ELA to pflacco, the following script can be executed. It does everyhing for you:
```bash
sh install.sh
```

Otherwise, please execute the following steps:
```bash
python -m pip uninstall pflacco --yes
python -m pip install -r requirements.txt
git clone https://github.com/Reiyan/pflacco.git
cp -rf MANIFEST.in pflacco/
cp -rf setup.py pflacco/
cp -r deep_ela pflacco/pflacco
cd pflacco
python -m pip install .
```

## Quickstart
Small, extended example from the pflacco package:
```python
from pflacco.sampling import create_initial_sample

from pflacco.classical_ela_features import calculate_ela_distribution
from pflacco.misc_features import calculate_fitness_distance_correlation

# This is new
from pflacco.deep_ela import load_medium_50d_v1, load_large_50d_v1
model = load_medium_50d_v1()

# The large models were published with stride=2, which keeps only every second token after
# the kNN embedding. stride=1 is RECOMMENDED for them: it roughly halves the feature noise
# and the sample size needed, at about 2.5x the runtime. See "Inference settings" below.
model_large = load_large_50d_v1(stride=1)

# Arbitrary objective function
def objective_function_1(x):
    return x[0]**2 - x[1]**2

# It also accepts multi-objective problems. So, let's define a second objective
def objective_function_2(x):
    return x[0]**0.5 - x[1]

dim = 2
# Create inital sample using latin hyper cube sampling
X = create_initial_sample(dim, sample_type = 'lhs')
# Calculate the objective values of the initial sample using an arbitrary objective function (here y = x1^2 - x2^2)
y = X.apply(lambda x: objective_function_1(x), axis = 1)

# Compute an exemplary feature set from the convential ELA features of the R-package flacco
ela_distr = calculate_ela_distribution(X, y)
print(ela_distr)

# Compute an exemplary feature set from the novel features which are not part of the R-package flacco yet.
fdc = calculate_fitness_distance_correlation(X, y)
print(fdc)

# Compute an exemplary feature set from Deep-ELA 
fdc = model(X, y, include_costs=True)
print(fdc)

# Now, let's do the same but with an bi-objective problem:
y_mo = X.apply(lambda x: [objective_function_1(x), objective_function_2(x)], axis = 1, result_type = 'expand')
fdc = model(X, y_mo, include_costs=True)
print(fdc)
```

## Inference settings

Two loader arguments change the quality of the features. Both default to the value
each checkpoint was published with, so nothing changes unless you ask for it — but
**`stride=1` is recommended for the large models**, which ship with `stride=2`.

Set them once on the model:

```python
model = load_large_50d_v1()                     # stride=2, as published
model = load_large_50d_v1(stride=1)             # recommended: better features, ~2.5x slower
model = load_large_50d_v1(stride=None)          # whatever the checkpoint's hparams.yaml says
model = load_medium_25d_v1(repetitions=50)      # reproducible to three decimals
```

...or override either for a single call, without rebuilding the model — the quick
way to see what a setting costs on your own problems:

```python
features = model(X, y)                          # the model's own settings
features = model(X, y, stride=2)                # this call only
features = model(X, y, stride=2, repetitions=50, include_costs=True)
```

An override never sticks: the next call goes back to the model's own values.
`predict()` takes the same `stride` and `repetitions` arguments.

### `stride` — defaults to the published value; use 1 on the large models

`forward` keeps every `stride`-th token after the kNN embedding, so a stride-2 model
attends over half its points. The published checkpoints disagree: `medium_*` ship
`stride: 1`, `large_*` ship `stride: 2`, and the loaders keep those defaults so
existing results reproduce. Measured against stride 1, stride 2 costs

- **~2x the feature noise** — resample-to-between-function distance goes 0.101 → 0.200
  (`medium_25d`) and 0.157 → 0.104 the other way for `large_25d`;
- **~2x the sample budget** — convergence needs ppd 20–25 at stride 1 against ppd 50
  at stride 2, and `large_25d_v1` at its shipped stride never reached stable features
  at any budget tested;
- **discriminability below about 50 points per dimension** — a 0.09–0.13 drop in a
  1-NN BBOB-group probe at 20 points per dimension, nothing at 100.

It buys about 2.5x throughput, which is the trade the published default makes. Unless
throughput is the binding constraint, pass `stride=1` to the large models.

At matched stride the four variants score the same, so `medium_25d_v1` — already
stride 1 — is the cheapest sensible default and needs no argument at all.

### `repetitions` — default 10

`predict` averages over random permutations of the `num_df` slots. Feature quality
saturates by about 10; past that it only buys reproducibility, with the run-to-run
jitter falling as `1/sqrt(repetitions)`. Use 50 or more when a number has to
reproduce. `EncoderBackbone.predict` still defaults to 1, which leaves roughly 0.04
of jitter between two runs of identical code.

### Sample size

Budgets scale with the dimension, matching the `25d` / `50d` in the model names:
features converge from about **20–25 points per dimension** at stride 1. That is 100–125
points at dim 5 but 200–250 at dim 10, so an absolute-point rule of thumb
under-specifies as the dimension grows.

### Do not pre-process

`KNNEmbedding` standardizes per sample and clamps to ±10 internally. Pass raw
coordinates and raw objective values; handing it normalized input is a silent error,
not an exception.

## Citation
If you are using Deep-ELA in any capacity, please use the following bibtex:
```
@misc{seiler2024deepela,
      title={Deep-ELA: Deep Exploratory Landscape Analysis with Self-Supervised Pretrained Transformers for Single- and Multi-Objective Continuous Optimization Problems}, 
      author={Moritz Vinzent Seiler and Pascal Kerschke and Heike Trautmann},
      year={2024},
      eprint={2401.01192},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```