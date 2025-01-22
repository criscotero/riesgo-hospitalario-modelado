import os
from pathlib import Path

DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset")
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

DATASET_MHAS_C2 = str(Path(DATASET_ROOT_PATH) / "H_MHAS_c2.sas7bdat")
DATASET_MHAS_URL = (
    "https://drive.google.com/uc?id=1PZRLL7cq6UAVLG3DFeuPuhrS1LJTsNvY&confirm=t"
)