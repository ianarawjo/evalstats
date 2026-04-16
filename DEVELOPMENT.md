# Development

## Building and validating the package before upload

Use this checklist before publishing to PyPI:

```bash
# from repository root
python -m pip install -U build twine
python -m build
python -m twine check dist/*
```

For a clean install smoke test from the built wheel:

```bash
python -m venv .pkgtest-venv
. .pkgtest-venv/bin/activate
python -m pip install -U pip
python -m pip install dist/evalstats-*.whl
python -c "import evalstats as p; print(p.__version__)"
evalstats --help
```

For runtime checks:

```bash
python -m pip install -e .[dev]
pytest -q
```

To test optional extras as well:

```bash
python -m pip install -e .[all]
pytest -q
```
