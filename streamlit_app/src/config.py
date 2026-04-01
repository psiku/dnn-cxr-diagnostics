from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
KEDRO_ROOT = ROOT_DIR / "cxr-training-pipeline"
KEDRO_SRC_DIR = KEDRO_ROOT / "src"
STREAMLIT_ROOT = ROOT_DIR / "streamlit_app"

for path in (ROOT_DIR, KEDRO_ROOT, KEDRO_SRC_DIR, STREAMLIT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

DATA_DIR = KEDRO_ROOT / "data"
REPORTING_DIR = DATA_DIR / "08_reporting"
BEST_THRESHOLDS_PATH = REPORTING_DIR / "best_thresholds.json"
IMAGES_DIR = DATA_DIR / "01_raw" / "images"
TRAIN_VAL_CSV = DATA_DIR / "02_intermediate" / "dataset_splits" / "train_val.csv"
TEST_CSV = DATA_DIR / "02_intermediate" / "dataset_splits" / "test.csv"

CKPT_PATH = KEDRO_ROOT / "checkpoints" / "best-epoch=11-val_ap=0.0000.ckpt"
