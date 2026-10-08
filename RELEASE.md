# GLiNER2 release guide

GLiNER2 supports Python 3.10–3.12. Releases are built from tags and published
with PyPI trusted publishing; maintainers must not upload artifacts with local
API tokens.

## One-time trusted-publishing setup

1. Create and protect a GitHub environment named `pypi`.
2. In the PyPI project settings, add this repository and the
   `.github/workflows/release.yml` workflow as a trusted publisher.
3. Add the repository variable `PYPI_TRUSTED_PUBLISHING_ENABLED=true`.

The publish job is skipped unless that variable is exactly `true`. Keep it
unset while validating the release workflow.

## Mandatory release gates

- [ ] Version in `gliner2/__init__.py` and the changelog release heading match.
- [ ] CI passes on Python 3.10, 3.11, and 3.12.
- [ ] Ruff, the incremental mypy target, and the coverage floor pass.
- [ ] Wheel and sdist pass `twine check` and artifact-content validation.
- [ ] Fresh base wheel and sdist installs remain torch-free.
- [ ] Fresh `[local]` and `[train]` wheel installs pass their smoke tests.
- [ ] Offline tests pass with no unexpected failure, skip, or xfail:

  ```bash
  pytest -m "not slow and not quality and not compile"
  ```

- [ ] Compile tests pass separately.
- [ ] The manual/nightly quality workflow passes with both exact checkpoints:
  - span: `fastino/gliner2-base-v1`
  - boundary: `fastino/gliner2.5-multi-v1`
- [ ] CUDA no-host-sync and performance checks pass on the self-hosted GPU
  runner. CPU and MPS checks do not substitute for this gate.
- [ ] Every hardware-only skip is recorded in the release notes.

## Reproduce artifact checks locally

```bash
python -m pip install ".[dev]"
python -m build
python -m twine check dist/*
python packaging_tests/check_artifact.py dist/*.whl
python packaging_tests/check_artifact.py dist/*.tar.gz
```

Run each install in a new virtual environment; do not reuse the build
environment:

```bash
python -m venv .smoke/base-wheel
.smoke/base-wheel/bin/python -m pip install dist/*.whl
.smoke/base-wheel/bin/python packaging_tests/smoke_install.py --profile base

python -m venv .smoke/local-wheel
wheel=$(printf '%s\n' dist/*.whl)
.smoke/local-wheel/bin/python -m pip install "${wheel}[local]"
.smoke/local-wheel/bin/python packaging_tests/smoke_install.py --profile local

python -m venv .smoke/train-wheel
.smoke/train-wheel/bin/python -m pip install "${wheel}[train]"
.smoke/train-wheel/bin/python packaging_tests/smoke_install.py --profile train

python -m venv .smoke/base-sdist
.smoke/base-sdist/bin/python -m pip install dist/*.tar.gz
.smoke/base-sdist/bin/python packaging_tests/smoke_install.py --profile base
```

## Tag and publish

1. Replace `Unreleased` on the target changelog heading with the release date.
2. Confirm the working tree and release branch contain only reviewed changes.
3. Create an annotated `vX.Y.Z` tag whose value exactly matches
   `gliner2.__version__`, then push the tag.
4. The release workflow rebuilds artifacts, validates them, and uploads them as
   a workflow artifact.
5. Only when trusted publishing is configured and
   `PYPI_TRUSTED_PUBLISHING_ENABLED=true` will the guarded publish job request
   an OIDC token and upload to PyPI.
6. Create the GitHub release from the validated tag and attach the workflow
   artifacts.

PyPI artifacts are immutable. If a version already exists, increment the
version and create a new tag; never replace or mutate an existing release.