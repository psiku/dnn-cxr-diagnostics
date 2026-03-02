# dnn-cxr-diagnostics

Deep Learning-based Chest X-Ray Diagnostics System. A bachelor's thesis project
featuring a CNN classifier for detecting pulmonary diseases, with a REST API and
placeholder for a web / desktop application for real-time clinical decision support.

## Project Structure

```
dnn-cxr-diagnostics/
├── src/dnn_cxr_diagnostics/   # Core Python package
│   ├── models/                # CNN architectures (ResNet-50 backbone)
│   ├── training/              # Training loops and helpers
│   ├── data/                  # Dataset classes and transforms
│   └── utils/                 # Visualisation and metrics utilities
├── notebooks/                 # Jupyter notebooks for EDA and model analysis
├── conf/                      # Kedro-ready configuration (catalog, parameters)
├── data/                      # Raw / interim / processed data (git-ignored)
├── models/                    # Saved model checkpoints (git-ignored)
├── app/                       # Web / desktop application
│   ├── main.py                # FastAPI application factory
│   ├── api/routes.py          # REST endpoints (/api/v1/predict)
│   └── ui/                    # Frontend placeholder (web SPA or desktop GUI)
├── tests/                     # pytest test suite
└── docs/                      # Architecture documentation
```

See [`docs/architecture.md`](docs/architecture.md) for a detailed overview.

## Quick Start

```bash
# Install the package in editable mode with all extras
pip install -e ".[dev,notebooks,app]"

# Run tests
pytest

# Launch the REST API
uvicorn app.main:app --reload

# Open a notebook
jupyter lab notebooks/
```

## Kedro

The `conf/` directory already follows Kedro conventions. To migrate to a full
Kedro project in the future, run `kedro new` and point it at the existing
`src/`, `conf/`, and `data/` directories.
