# Comparing historical vs. corrected t-momentum

This archive provides the modernized optimizer code while preserving the
historical Student-t and TAdamW update rules used by the May 2023 repository.

## Three meaningful versions

1. **Unmodified source** (original GitHub commit): original behavior and old APIs;
   essential archival reference.
2. **Compatibility-only / old behavior** (this ZIP): updated closures, dtype
   handling, older-API cleanups, and tests, while retaining historical numerical
   behavior for optimizers that originally ran.
3. **Compatibility + algorithmic corrections** (the separately supplied
   `tmomentum_modernized_repo.zip`): also persists W_t, z_ave and z_var, and
   implements standard AdamW epsilon/bias-correction placement.

## Testing

From the extracted folder:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

A parity test compares 44 historical reference cases against outputs saved
from the original May 2023 archive using deterministic gradient sequences.
These are short CPU tests rather than a proof of exact equality across all
PyTorch versions, floating-point dtypes, or training workloads.

For fair optimizer-quality experiments, keep the same data order, initialization,
seeds, hardware, and hyperparameters. Compare separately: fixed DoF (`beta_dof=1`),
adaptive DoF (`beta_dof<1`), and TAdamW, as these reveal different corrections.
Use separate environments to avoid conflicting editable installations.

`TRAdam`/`TPlainRAdam` with finite `k_dof` crashed at initialization in the
original code. Here their initialization is fixed. Their results therefore
cannot be called historical parity, even though the downstream Student-t
recurrence is intentionally preserved.

## Git strategy

- Tag the *unmodified* main commit as `v1.0.2-legacy` before applying changes.
- Commit this archive on a separate `experiments/legacy-behavior` branch.
- Commit the corrected version on a separate `modernization` branch.
- After experiments, merge or release the selected version with explicit notes.

This ZIP is an experimental source archive, not a PyPI release.
