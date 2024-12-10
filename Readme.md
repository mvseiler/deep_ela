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
from pflacco.deep_ela import load_medium_50d_v1
model = load_medium_50d_v1()

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