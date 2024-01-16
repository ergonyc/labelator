## wrappers to load data
import anndata as ad
from scvi.model import SCVI, SCANVI
from pathlib import Path

from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from .models._lbl8r import LBL8R
from .utils import (
    make_latent_adata,
    # add_predictions_to_adata,
    # merge_into_obs,
    # query_scanvi,
    # plot_scvi_training,
    # plot_scanvi_training,
    # plot_lbl8r_training,
    make_pc_loading_adata,
)

from .constants import *
from .modules._xgb import train_xgboost, test_xgboost, load_xgboost, get_xgb_data
from ._artifacts import Adata

"""

Loading should

0. probe the path to see if an "out" version exists
    - if so, load it and return it
    - if not, load the raw data and prep it

1. defone the data paths
    - input.
    - outut. 
    - exptra.  e.g. expr version

2. prep the data.
2. load the data



"""


# wrap adata into a dataclass
def load_adata(
    adata_path: str | Path,
) -> Adata:
    """
    Load data for training wrapped in Adata for easy artifact generation
    """
    adata_path = Path(adata_path)
    return Adata(adata_path)


def load_training_data(adata_path: str | Path) -> Adata:
    """
    Load training data.
    """
    adata_path = Path(adata_path)
    # warn if "train" is not in data_path.name
    if "train" not in adata_path.name:
        print(
            f"WARNING:  'train' not in data_path.name: {adata_path.name}.  "
            "This may cause problems with the output file names."
        )

    return Adata(adata_path)


def load_query_data(adata_path: str | Path) -> Adata:
    """
    Load query data.
    """

    adata_path = Path(adata_path)
    # warn if "train" is not in data_path.name
    if "test" not in adata_path.name:
        print(f"WARNING:  'train' not in data_path.name: {adata_path.name}.  ")

    return Adata(adata_path)
