# t-momentum
## A Stochastic Gradient momentum based on a Student-t distribution Exponential Moving Average

 Official repository for the t-momentum algorithm.

**Journal Paper**: *Robust Stochastic Gradient Descent with Student-t Distribution based First-order Momentum* (Accepted for publication in the IEEE Transactions on Neural Networks and Learning Systems journal).

**Arxiv Preprint** (early version. Focuses only on the integration of the t-momentum to Adam. Corresponding repository [here](https://github.com/Mahoumaru/TAdam)): [*TAdam: A Robust Stochastic Gradient Optimizer*](http://arxiv.org/abs/2003.00179)
 
## How to use:

1. Install with pip
```
git clone https://github.com/Mahoumaru/t-momentum.git
cd t-momentum
pip install -e .
```
2. Import and use each optimizer just like you would use an official pytorch optimizer (adjust hyperparameters such as learning rate, k_dof, betas, weight_decay, amsgrad, etc.)
```python
from tmomentum.optimizers.TAdam import TAdam
from tmomentum.optimizers.TYogi import TYogi

optimizer1 = TAdam(net1.parameters())
optimizer2 = TYogi(net2.parameters())
```

## How to cite:
 If you employ the t-momentum based optimizers in your Machine Learning application, please cite us using the following:

**Journal Paper**

To come.

**Arxiv Preprint**
```
@article{ilboudo2020tadam,
  title={TAdam: A Robust Stochastic Gradient Optimizer},
  author={Ilboudo, Wendyam Eric Lionel and Kobayashi, Taisuke and Sugimoto, Kenji},
  journal={arXiv preprint arXiv:2003.00179},
  year={2020}
}
```

## Note
 This repository is implemented in pytorch.
 A tensorflow implementation of the t-momentum integrated to various optimizers would be really appreciated. Don't hesitate to PR.
