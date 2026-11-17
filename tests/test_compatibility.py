"""Compatibility and historical-behavior checks for tmomentum.

Run from the repository root:
    python -m unittest discover -s tests -v

These checks intentionally preserve the historical numerical updates.
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

    def test_tadamw_historical_epsilon_order(self):
        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        op = optim.TAdamW([p], lr=0.1, weight_decay=0, eps=1e-8,
                          k_dof=math.inf)
        g = 1e-9
        p.grad = torch.tensor([g], dtype=torch.float64)
        op.step()
        beta1, beta2 = 0.9, 0.999
        m = (1-beta1)*g
        v = (1-beta2)*g*g
        step_size = 0.1 * math.sqrt(1-beta2) / (1-beta1)
        old_expected = 1.0 - step_size*m/(math.sqrt(v)+1e-8)
        self.assertAlmostEqual(p.item(), old_expected, places=12)

    def test_historical_adaptive_dof_statistics_remain_initial(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        op = optim.TAdam([p], k_dof=1.0, beta_dof=0.9)
        for g in ([0.1, 0.2], [0.3, -0.7], [0.8, -0.1]):
            p.grad = torch.tensor(g)
            op.step()
        self.assertEqual(op.state[p]["z_ave"], 0.0)
        expected = 2.0*(1.0+2.0)/(1.0**2) + op.state[p]['trigamma_value']
        self.assertEqual(op.state[p]["z_var"], expected)
        self.assertIn('dof', op.state[p])

    def test_historical_student_t_accumulator_is_constant(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        op = optim.TAdam([p], k_dof=1.0, beta_dof=1.0)
        p.grad = torch.tensor([0.1, 0.2])
        op.step()
        first = op.state[p]['W_t']
        p.grad = torch.tensor([-0.3, 0.4])
        op.step()
        self.assertEqual(op.state[p]['W_t'], first)
        self.assertAlmostEqual(first, 0.9 / 0.1)

    def test_tadamw_historical_epsilon_order_multistep(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -0.2], dtype=torch.float64))
        op = optim.TAdamW([p], lr=0.003, weight_decay=0, eps=1e-3,
                          k_dof=math.inf)
        baseline = p.detach().clone()
        m = torch.zeros_like(p)
        v = torch.zeros_like(p)
        beta1, beta2 = 0.9, 0.999
        for t in range(1, 13):
            g = torch.tensor([1e-5*t, -0.8], dtype=torch.float64)
            p.grad = g.clone()
            op.step()
            m = beta1*m + (1-beta1)*g
            v = beta2*v + (1-beta2)*g.square()
            step_size = 0.003*math.sqrt(1-beta2**t)/(1-beta1**t)
            baseline = baseline - step_size*m/(torch.sqrt(v)+1e-3)
        self.assertTrue(torch.allclose(p, baseline, atol=1e-12, rtol=0))

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


class HistoricalParity(unittest.TestCase):
    def test_reference_2023_trajectories(self):
        """Compare with checkpoints generated from unmodified May 2023 source.

        Finite-k_dof TRAdam variants crashed in that source and are excluded
        from the reference data; the modernization fixes their first step.
        """
        import json
        from pathlib import Path
        fixture = Path(__file__).resolve().parent / 'fixtures' / 'historical_outputs.json'
        with fixture.open(encoding='utf-8') as file:
            expected = json.load(file)
        grads = ([0.003, -0.25], [-0.33, 0.7], [0.002, -0.004],
                 [0.3, -0.1], [-0.4, 0.9])
        for case, snapshots in expected.items():
            name, dof_str, beta_dof_str = case.split('|')
            p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float32))
            op = getattr(optim, name)([p], lr=0.003, weight_decay=0.,
                                      k_dof=float(dof_str), beta_dof=float(beta_dof_str))
            with self.subTest(case=case):
                for i, g in enumerate(grads):
                    p.grad = torch.tensor(g, dtype=p.dtype)
                    op.step()
                    actual = op.state[p]
                    ref = snapshots[i]
                    for result, target in zip(p.detach().tolist(), ref['param']):
                        self.assertAlmostEqual(result, target, delta=2e-6)
                    for state_key in ('step', 'W_t', 'dof', 'z_ave', 'z_var'):
                        if state_key in ref:
                            value = actual[state_key]
                            if torch.is_tensor(value):
                                value = value.item()
                            self.assertAlmostEqual(float(value), ref[state_key],
                                                   delta=1e-5 + 1e-6*abs(ref[state_key]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
