# Architecture Overview

## Directory Layout

```
dnn-cxr-diagnostics/
│
├── src/dnn_cxr_diagnostics/   # Core Python package
│   ├── models/                # CNN model definitions
│   ├── training/              # Training loops & helpers
│   ├── data/                  # Dataset classes & transforms
│   └── utils/                 # Visualisation & metrics
│
├── notebooks/                 # Jupyter notebooks for EDA & analysis
│   ├── 01_data_exploration.ipynb
│   └── 02_model_analysis.ipynb
│
├── conf/                      # Kedro-ready configuration
│   ├── base/
│   │   ├── catalog.yml        # Dataset catalog
│   │   └── parameters.yml     # Hyper-parameters
│   └── local/                 # Local overrides (git-ignored)
│
├── data/                      # Data directories (contents git-ignored)
│   ├── raw/                   # Original, unmodified images
│   ├── interim/               # Intermediate processed data
│   └── processed/             # Train / val / test splits
│
├── models/                    # Saved model checkpoints (git-ignored)
│
├── app/                       # Web / desktop application
│   ├── main.py                # FastAPI application factory
│   ├── api/
│   │   └── routes.py          # REST endpoints
│   └── ui/                    # Frontend (web SPA or desktop GUI)
│
├── tests/                     # pytest test suite
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
│
└── docs/                      # Project documentation
    └── architecture.md        # This file
```

## Key Design Decisions

### `src/` layout
The package lives under `src/` to enforce clean separation between the
installable library and the rest of the project.  This is also the layout
expected by Kedro when the project is later migrated.

### Kedro-readiness (`conf/`)
`conf/base/catalog.yml` and `conf/base/parameters.yml` follow Kedro
conventions so that the training pipeline can be wrapped in a Kedro project
with minimal refactoring.

### Application layer (`app/`)
A FastAPI REST service exposes `/api/v1/predict` for integration with any
frontend.  The `app/ui/` directory is reserved for a web (React/Vue) or
desktop (PyQt6/Tkinter) GUI.

## Data Flow

```
data/raw  →  data/interim  →  data/processed/{train,val,test}
                                        ↓
                              src/.../training/train.py
                                        ↓
                                  models/best_model.pt
                                        ↓
                              app/api/routes.py  →  HTTP response
```
