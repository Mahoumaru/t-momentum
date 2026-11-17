# Proposed compatibility/correctness patch (not an upstream release)

Based on the May 2023 source archive supplied by the repository owner.

## Behavioral corrections (can change trajectories)

- The Student-t accumulated weight `W_t` is now persisted every step. Previously
  the float in the optimizer state never changed after initialization.
- Adaptive-DoF running statistics `z_ave` and `z_var` are now persisted along
  with the resulting degrees of freedom. This affects runs with `beta_dof < 1`.
- `TRAdam`/`TPlainRAdam` increment their step before computing t-momentum,
  fixing missing `W_t` initialization on the first step.
- `TAdamW` uses the AdamW bias-corrected second moment and epsilon placement
  so `k_dof=inf` recovers `torch.optim.AdamW` (for the same lr schedule).
- `TPAdam` no longer changes the caller's `.grad` when weight_decay is nonzero.

## API/maintenance corrections

- Closures execute under `torch.enable_grad()`, while parameter updates use
  `torch.no_grad()`.
- Remove deprecated positional `Tensor.add_` overloads in TAdamW.
- Remove `.data` from TAdamW, TRAdam, TPlainRAdam, TPAdam; preserve FP64
  rather than unconditionally converting it to FP32 in the first three.
- Fix the invalid no-argument construction of the standalone TMomentum helper.

## Compatibility and reproducibility notes

- Keeps constructor arguments and existing optimizer state keys.
- Old checkpoints are *loadable* subject to standard PyTorch restrictions,
  but the corrected recurrences can change results after resuming.
- Tested locally against PyTorch 2.10.0+cpu on Python 3.13.
  Older versions, CUDA, AMP, compilation and graph capture are **not validated**.
- Run the included tests: `python -m unittest discover -s tests -v`.
- This patch is not a faithful bit-for-bit continuation of the 2023 behavior;
  keep a tagged legacy release for exact reproduction of published runs.
