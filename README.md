# t-momentum
## A Stochastic Gradient momentum based on a Student-t distribution Exponential Moving Average

 Official repository for the t-momentum algorithm.

**Journal Paper** (Accepted for publication in the IEEE Transactions on Neural Networks and Learning Systems journal):
[*Robust Stochastic Gradient Descent With Student-t Distribution Based First-Order Momentum*](https://ieeexplore.ieee.org/document/9296551)

**Arxiv Preprint** (early version. Focuses only on the integration of the t-momentum to Adam. Corresponding repository [here](https://github.com/Mahoumaru/TAdam)): [*TAdam: A Robust Stochastic Gradient Optimizer*](http://arxiv.org/abs/2003.00179)

## How to use:

1. Install
- install with pip:
```
pip install tmomentum
```
- or clone and install:
```
git clone https://github.com/Mahoumaru/t-momentum.git
cd t-momentum
pip install -e .
```

2. Import and use each optimizer just like you would use an official pytorch optimizer (adjust hyperparameters such as learning rate, k_dof, betas, weight_decay, amsgrad, etc.)
```python
from tmomentum.optimizers import TAdam
from tmomentum.optimizers import TYogi

optimizer1 = TAdam(net1.parameters())
optimizer2 = TYogi(net2.parameters())
```

## How to cite:
 If you employ the t-momentum based optimizers in your Machine Learning application, please cite us using the following:

##### Plain Text
```
W. E. L. Ilboudo, T. Kobayashi and K. Sugimoto,
"Robust Stochastic Gradient Descent With Student-t Distribution Based First-Order Momentum,"
in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2020.3041755.
```

##### Bibtex
```
@ARTICLE{9296551,
  author={W. E. L. {Ilboudo} and T. {Kobayashi} and K. {Sugimoto}},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Robust Stochastic Gradient Descent With Student-t Distribution Based First-Order Momentum},
  year={2020},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2020.3041755}}
```

## Note
 This repository is implemented in pytorch.
 A tensorflow implementation of the t-momentum integrated to various optimizers would be really appreciated. Don't hesitate to PR.
