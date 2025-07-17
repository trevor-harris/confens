# Conformal Ensembles for Constrained Climate UQ

Paper: [Quantifying uncertainty in climate projections with conformal ensembles](https://arxiv.org/abs/2408.06642)

## Overview

`confensembles` implements the conformal ensembling approach (CE) [1] in JAX along with the baseline model analysis functions [2]. 

Conformal ensembling (CE) uses conformal prediction sets and observational data to constrain projection uncertainty in General Circulation Model (GCM) ensembles. This approach works by first training a model analysis function, such as a Convolutional Neural Network, to use the GCM ensemble to ``predict'' observational data. Then, on held out data we compute the prediction residuals. These residuals are treated as functional data so that we can employ functional data depth techniques to define exact prediction sets in function space with statistical guarantees. 

## Installation
```bash
git clone https://github.com/trevor-harris/confens
pip install confens/
```

## Examples

```python
import numpy as np
import conformal_ensembles as ce

# significance level
alpha = 0.1

# calibration data
np.random.seed(1023)
y_cal = np.random.randn(500, 30, 30)
yhat_cal = np.random.randn(500, 30, 30)

# test data
y_test = np.random.randn(500, 30, 30)
yhat_test = np.random.randn(500, 30, 30)

# compute residuals
res_val = y_cal - yhat_cal
res_test = y_test - yhat_test

# cutoff value
q_val = ce.conf_quantile(res_val, alpha)
depth_test = ce.tukey_depth(res_val, res_test)
np.mean(depth_test >= q_val) # 0.90200

# full ensemble
ens_val = ce.conf_ensemble(res_val, alpha)
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


