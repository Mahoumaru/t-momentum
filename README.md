# t-momentum
## A Stochastic Gradient momentum based on a Student-t distribution Exponential Moving Average

 Official repository for the t-momentum algorithm.
 
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

## Note
 This repository is implemented in pytorch.
 A tensorflow implementation of the t-momentum integrated to various optimizers would be really appreciated. Don't hesitate to PR.
