## wrappers for getting trained models
# mac_load / mac_get random code
"""
strategy:
- create all the paths for models and artifacts at setup.

0. create the paths, set the args for the pipeline

Pipeline:
1. load the "input" data
2. get the trained model
3. query the model
4. create the artifacts
    - figures
    - models
    - auxiliary data
        - latent versions
        - pcs version
        - scvi normalized version
    - predictions updated metadata (tabular)
    - vars (genes)



Pipeline 2:  (query pre-existing model)
- scanvi model:
    - load the model
        - create auxilarry data:
    - create/fit surgery model
    - query the surgery model
    - create artifacts:
        - figures
        - models
        - auxiliary data
            - latent versions
            - pcs version
            - scvi normalized version
        - predictions updated metadata
        - vars (genes) list of genes used for training, maybe include variance?
        

artifacts:
- figures
- predcitions (tables)
- latents (adata)

TODO: 
    1. simplify "dataprep" routines. i.e. don't add all the metadat, just put it somewhere
    2. make stubs for .var diversity.  i.e. reference to the 3k genes being trained on
        - how to deal with "empty" genes in query data?
    3. simplify artifacts to save PCs, naming convention for table artifacts (obs) of predictions 
        - parquet vs. csv?
    4. cli (click) interface
    5 helpers to re-aggregate tabular data with adata for easy visulaization


"""

from anndata import AnnData
from scvi.model import SCVI, SCANVI
from pathlib import Path
from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from .model._lbl8r import (
    LBL8R,
    get_lbl8r,
    query_lbl8r,
    get_scvi_lbl8r,
    get_pca_lbl8r,
)

from .model._scvi import (
    get_trained_scvi,
    get_trained_scanvi,
    get_query_scvi,
    get_query_scanvi,
    query_scanvi,
)

from .model._xgb import get_xgb, query_xgb
from .model.utils._data import Adata
from .model.utils._Model import Model
from ._constants import *

PRED_KEY = "label"
INSERT_KEY = "pred"
# Model = SCVI | SCANVI | LBL8R | Booster


def setup_paths(model_params_file, data_path, config_path):
    """
    Create the paths for models and artifacts
    """

    # load the model params
    model_params = yaml.load(model_params_file)

    # create the paths
    model_path = Path(model_params["model_path"])
    model_path.mkdir(exist_ok=True)

    # create the data paths
    data_path = Path(data_path)
    data_path.mkdir(exist_ok=True)

    # create the config paths
    config_path = Path(config_path)
    config_path.mkdir(exist_ok=True)

    # create the paths for the artifacts

    # create the paths for the models

    # create the paths for the data

    # create the paths for the config


def load_and_prep(data_path):
    """
    Load and prep the data
    """

    # prep

    load_adata(data_path)


def create_artifacts(visualization_path, artifacts_path):
    """
    Create the artifacts
    """

    # create the artifacts
    if visualization_path:
        # create the visualizations
        pass

    if artifacts_path:
        # create the artifacts
        pass


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
3. load the data
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


def get_model(
    data: Adata,
    model_name: str,
    model_path: Path | str,
    model: SCVI | SCANVI | LBL8R | None = None,
    labels_key: str = "cell_type",
    # batch_key: str | None = None, # TODO:  model_kwargs for controlling batch_key, etc..
    retrain: bool = False,
    **training_kwargs,
) -> (Model, Adata):
    """
    Load a model from the model_path

    Parameters
    ----------
    model_name : str
        Name of the model.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.

    Returns
    -------
    SCVI | SCANVI | LBL8R | Booster
        A model object.

    """

    model_path = model_path / model_name
    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    if model_name.startswith("scvi"):
        _get_model = get_trained_scvi

    elif model_name.startswith("scanvi"):
        _get_model = get_trained_scanvi

    elif model_name.startswith("lbl8r"):
        if model_name == "lbl8r_pca":
            # do dataprep
            _get_model = get_pca_lbl8r
        elif model_name == "lbl8r_scvi":
            # do dataprep
            _get_model = get_scvi_lbl8r
        else:
            _get_model = get_lbl8r

    elif model_name.startswith("xgb"):
        _get_model = get_xgb

    elif model_name.startswith("query_scvi"):
        _get_model = get_query_scvi

    elif model_name.startswith("query_scanvi"):
        _get_model = get_query_scanvi

    else:
        raise ValueError(f"unknown model_name {model_name}")

    model, ad = _get_model(
        data.adata,
        labels_key=labels_key,
        model_path=model_path,
        retrain=retrain,
        model_name=model_name,
        **training_kwargs,
    )

    # update data with ad
    data.update(ad)

    return model


def query_model(
    adata: AnnData,
    model: LBL8R | SCANVI | Booster,
    label_encoder: LabelEncoder | None = None,
    labels_key: str = "cell_type",
) -> AnnData:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model : LBL8R | SCANVI | Booster
        An classification model.
    label_encoder : LabelEncoder
        The label encoder.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    if isinstance(model, LBL8R):
        _quiery_model = query_lbl8r
    elif isinstance(model, SCANVI):
        _quiery_model = query_scanvi
    elif isinstance(model, Booster):
        _quiery_model = query_xgb

    else:
        raise ValueError(f"model {model} is not a valid model")

    ad = _quiery_model(adata, model, labels_key=labels_key)

    # update data with ad
    data.update(ad)

    return model
