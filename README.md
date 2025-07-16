# ConfEns

Conformal Ensembles for Climate UQ.

[Quantifying uncertainty in climate projections with conformal ensembles](https://arxiv.org/abs/2408.06642)

## Overview

`ConfEns` implements the conformal ensembling approach (CE) [1] in JAX along with the baseline model analysis functions [2]. 

Conformal ensembling (CE) uses conformal prediction sets and observational data to constrain projection uncertainty in General Circulation Model (GCM) ensembles. This approach works by first training a model analysis function, such as a Convolutional Neural Network, to use the GCM ensemble to ``predict'' observational data. Then, on held out data we compute the prediction residuals. These residuals are treated as functional data so that we can employ functional data depth techniques to define exact prediction sets in function space with statistical guarantees. 

## Installation
```bash
git clone https://github.com/trevor-harris/confens
pip install confens/
```

## Examples

```python
import torch
import torch_harmonics as th
from torch_harmonics.random_fields import GaussianRandomFieldS2
from scwd.metrics import scwd

# generate two GPs on the sphere (nlat = 90, nlon = 180)
GRF_x = GaussianRandomFieldS2(nlat = 90)
GRF_y = GaussianRandomFieldS2(nlat = 90)

# Sample 100 fields for X, 200 for Y
x = GRF_x(100)
y = GRF_y(200)

scwd_map, scwd_val = scwd(x, y)
```

## Notes
This package is under active development. The scripts provided in /scripts are raw and completely unedited from the original submission. They may require a substantial amount of GPU RAM to run. Climate model data was acquired manually from the [Earth System Grid Federation](https://esgf.github.io). Scripts to automatically download and process climate model data will be added in the future.

## Cite us

If you use `ConfEns` in an academic paper, please cite [1]

```bibtex
@article{harris2024quantifying,
  title={Quantifying uncertainty in climate projections with conformal ensembles},
  author={Harris, Trevor and Sriver, Ryan},
  journal={arXiv preprint arXiv:2408.06642},
  year={2024}
}
```
## References
<a id='1'>[1]</a>
Harris T., Sriver, R.; 
Quantifying uncertainty in climate projections with conformal ensembles;
Annals of Applied Statistics (under review), 2024. [arxiv link](https://arxiv.org/abs/2401.14657)

<a id="1">[2]</a>
Harris T., Li B., Sriver, R.; 
Multimodel ensemble analysis with neural network Gaussian processes
Annals of Applied Statistics, 2023. [arxiv link](https://arxiv.org/abs/2202.04152)


