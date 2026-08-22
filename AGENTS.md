# Aurauv Engineering Contract

## Non-negotiable invariants

1. uv owns dependency resolution, lockfiles, Python installation, and environment synchronization.
2. Never repair a project environment with a post-sync `pip install` outside the uv lock.
3. Determine `invocation_root` and outer `environment_owner` before routing.
4. A managed workspace has one owner environment; never create a member-local `.venv`.
5. A project that is both an outer member and an inner workspace root is treated as an outer member.
6. Fallback requires explicit CLI authorization, interactive confirmation, or `--aura-yes`; configuration alone is not consent.
7. Write state only after uv, provider verification, and project import checks all succeed.
8. Package-specific platform logic belongs in providers, not the generic engine.
9. Preserve uv argument semantics. Parse `--aura-*` only before the uv command boundary.
10. Read-only commands must not update uv, install Python, refresh member locks, repair packages, or write state.

## Required validation for changes

Run:

```bash
python -m pytest
python -m compileall -q src tests
python scripts/build_zipapp.py --output /tmp/aurauv.pyz
python /tmp/aurauv.pyz aura version
```

Provider changes need contract, detection, preflight, fallback, runtime verification, state invalidation, and current-interpreter tests.

## Extension rule

For Triton, GUI/headless wheels, ROCm, XPU, or another mutually exclusive family, add a provider and normally a separate route. Do not expand accelerator profiles into a cross-product unless the package contracts are truly inseparable.
