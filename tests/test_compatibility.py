"""Compatibility/correctness regression checks for tmomentum.

Run from the repository root:
    python -m unittest discover -s tests -v

These are regression checks for the repaired implementation.
Designed to avoid modern-only PyTorch APIs where possible.
"""
import copy
import math
import unittest

import torch
from tmomentum import optimizers as optim
from tmomentum.utils.tmomentum import TMomentum


NAMES = (
    "TAdam", "TAdamW", "TEAdam", "TPAdam", "TSAdam", "TRAdam",
    "TPlainRAdam", "TAdaBound", "TAdaBelief", "TLaProp", "TDiffGrad", "TYogi",
)


def new_optimizer(name, parameter, **kwargs):
    return getattr(optim, name)([parameter], **kwargs)


class OptimizerCompatibility(unittest.TestCase):
    def test_basic_finite_and_infinite_dof(self):
        for name in NAMES:
            for k_dof in (1.0, math.inf):
                with self.subTest(name=name, k_dof=k_dof):
                    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
                    op = new_optimizer(name, p, k_dof=k_dof)
                    for _ in range(3):
                        p.grad = torch.tensor([0.25, -0.4])
                        op.step()
                    self.assertTrue(torch.isfinite(p).all().item())

    def test_closure_can_compute_gradients(self):
        for name in NAMES:
            with self.subTest(name=name):
                p = torch.nn.Parameter(torch.tensor(2.0))
                op = new_optimizer(name, p, k_dof=math.inf)

                def closure():
                    op.zero_grad()
                    loss = p.square()
                    loss.backward()
                    return loss

                op.step(closure)
                self.assertIsNotNone(p.grad)

    def test_checkpoint_continuation(self):
        for name in NAMES:
            with self.subTest(name=name):
                p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
                op = new_optimizer(name, p, k_dof=1.0)
                for _ in range(2):
                    p.grad = torch.tensor([0.5, -0.2])
                    op.step()
                q = torch.nn.Parameter(p.detach().clone())
                restored = new_optimizer(name, q, k_dof=1.0)
                restored.load_state_dict(copy.deepcopy(op.state_dict()))
                p.grad = torch.tensor([-0.1, 0.3])
                q.grad = p.grad.clone()
                op.step()
                restored.step()
                self.assertTrue(torch.allclose(p, q, rtol=1e-6, atol=1e-8))

    def test_tadam_infinite_dof_matches_pytorch_adam(self):
        for amsgrad in (False, True):
            with self.subTest(amsgrad=amsgrad):
                p = torch.nn.Parameter(torch.tensor([1.0, -0.2], dtype=torch.float64))
                q = torch.nn.Parameter(p.detach().clone())
                reference = torch.optim.Adam([p], lr=0.002, weight_decay=0.1, amsgrad=amsgrad)
                candidate = optim.TAdam([q], lr=0.002, weight_decay=0.1,
                                        amsgrad=amsgrad, k_dof=math.inf)
                for t in range(10):
                    g = torch.tensor([0.1 + 0.01*t, -0.7], dtype=torch.float64)
                    p.grad = g.clone()
                    q.grad = g.clone()
                    reference.step()
                    candidate.step()
                self.assertTrue(torch.allclose(p, q, rtol=0, atol=1e-12))

    def test_tadamw_infinite_dof_matches_pytorch_adamw(self):
        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        q = torch.nn.Parameter(p.detach().clone())
        reference = torch.optim.AdamW([p], lr=0.1, weight_decay=0, eps=1e-8)
        candidate = optim.TAdamW([q], lr=0.1, weight_decay=0,
                                 eps=1e-8, k_dof=math.inf)
        p.grad = torch.tensor([1e-9], dtype=torch.float64)
        q.grad = p.grad.clone()
        reference.step()
        candidate.step()
        self.assertTrue(torch.allclose(p, q, rtol=0, atol=1e-12),
                        msg="The epsilon/bias-correction placement differs")

    def test_adaptive_dof_running_statistics_change(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        op = optim.TAdam([p], k_dof=1.0, beta_dof=0.9)
        for g in ([0.1, 0.2], [0.3, -0.7], [0.8, -0.1]):
            p.grad = torch.tensor(g)
            op.step()
        self.assertNotEqual(op.state[p]["z_ave"], 0.0,
                            msg="z_ave changes only locally in update_dof")

    def test_student_t_accumulator_persists(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        op = optim.TAdam([p], k_dof=1.0, beta_dof=1.0)
        p.grad = torch.tensor([0.1, 0.2])
        op.step()
        first = op.state[p]['W_t']
        p.grad = torch.tensor([-0.3, 0.4])
        op.step()
        self.assertNotEqual(op.state[p]['W_t'], first)

    def test_tadamw_infinite_dof_matches_adamw_multistep(self):
        for weight_decay in (0.0, 0.1):
            for eps in (1e-8, 1e-3):
                with self.subTest(weight_decay=weight_decay, eps=eps):
                    p = torch.nn.Parameter(torch.tensor([1.0, -0.2], dtype=torch.float64))
                    q = torch.nn.Parameter(p.detach().clone())
                    reference = torch.optim.AdamW([p], lr=0.003, weight_decay=weight_decay,
                                                  eps=eps)
                    candidate = optim.TAdamW([q], lr=0.003, weight_decay=weight_decay,
                                             eps=eps, k_dof=math.inf)
                    for t in range(12):
                        g = torch.tensor([1e-5*(t+1), -0.8], dtype=torch.float64)
                        p.grad = g.clone()
                        q.grad = g.clone()
                        reference.step()
                        candidate.step()
                    self.assertTrue(torch.allclose(p, q, atol=1e-12, rtol=0),
                                    msg="TAdamW should agree with torch.optim.AdamW")

    def test_adaptive_dof_checkpoint_continuation(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        op = optim.TAdam([p], k_dof=1.0, beta_dof=0.9)
        for g in ([0.1, -0.2], [-0.5, 0.6], [0.3, 0.7]):
            p.grad = torch.tensor(g, dtype=p.dtype)
            op.step()
        q = torch.nn.Parameter(p.detach().clone())
        restored = optim.TAdam([q], k_dof=1.0, beta_dof=0.9)
        restored.load_state_dict(copy.deepcopy(op.state_dict()))
        for g in ([0.5, 0.1], [-0.2, -0.7]):
            p.grad = torch.tensor(g, dtype=p.dtype)
            q.grad = p.grad.clone()
            op.step()
            restored.step()
        self.assertTrue(torch.allclose(p, q, atol=1e-12, rtol=0))
        for key in ('W_t', 'dof', 'z_ave', 'z_var'):
            self.assertEqual(op.state[p][key], restored.state[q][key])

    def test_weight_decay_does_not_mutate_user_gradient(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        op = optim.TPAdam([p], weight_decay=0.1, k_dof=math.inf)
        p.grad = torch.tensor([0.1, 0.2])
        before = p.grad.clone()
        op.step()
        self.assertTrue(torch.equal(p.grad, before),
                        msg="TPAdam changes p.grad via grad.add_(p, ...)")

    def test_tmomentum_helper_constructible(self):
        self.assertIsInstance(TMomentum(), TMomentum)


if __name__ == "__main__":
    unittest.main(verbosity=2)
