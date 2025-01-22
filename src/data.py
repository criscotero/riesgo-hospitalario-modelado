import os
import gdown
import pandas as pd
import re

from src import config

def get_datasets(
        chunksize: int = 10000
) -> pd.DataFrame:
    """
    Downloads and loads SAS dataset from GDrive.

    :param chunksize: Dataset chunk size (avoids memory overload).
    :return: Raw dataframe with information of elderly people interviews.
    """
    # Download H_MHAS_c2.sas7bdat
    if not os.path.exists(config.DATASET_MHAS_C2):
        gdown.download(
            config.DATASET_MHAS_URL, config.DATASET_MHAS_C2, quiet=False
        )

    # read the dataset H_MHAS_c2 from the file (470 MB)
    file_path = os.path.join(config.DATASET_ROOT_PATH, config.DATASET_MHAS_C2)

    return pd.concat([
        df for df in pd.read_sas(file_path, chunksize=chunksize)
    ])