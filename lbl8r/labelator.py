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
    prep_pcs_adata,
)

from .model._scvi import (
    get_trained_scvi,
    get_trained_scanvi,
    get_query_scvi,
    get_query_scanvi,
    query_scanvi,
    prep_latent_z_adata,
)

from .model._xgb import get_xgb, query_xgb
from .model.utils._data import Adata
from .model.utils._Model import Model
from ._constants import *

MODEL = SCVI | SCANVI | LBL8R | Booster

PRED_KEY = "label"
INSERT_KEY = "pred"
# Model = SCVI | SCANVI | LBL8R | Booster

# SCANVI/SCVI model names
SCVI_SUB_MODEL_NAME = "scvi"
SCANVI_SUB_MODEL_NAME = "scanvi"
QUERY_SCVI_SUB_MODEL_NAME = "query_scvi"
QUERY_SCANVI_SUB_MODEL_NAME = "query_scanvi"
LBL8R_SCVI_SUB_MODEL_NAME = "scvi_emb"

# LBL8R model names
SCVI_LATENT_MODEL_NAME = "lbl8r_scvi_emb"
RAW_PC_MODEL_NAME = "lbl8r_raw_cnt_pcs"
SCVI_EXPR_PC_MODEL_NAME = "lbl8r_scvi_expr_pcs"

# LBL8R XGBoost model names
XGB_SCVI_LATENT_MODEL_NAME = "xgb_scvi_emb"
XGB_RAW_PC_MODEL_NAME = "xgb_raw_cnt_pcs"
XGB_SCVI_EXPR_PC_MODEL_NAME = "xgb_scvi_expr_pcs"

# E2E model names
# lbl8r
LBL8R_SCVI_EXPRESION_MODEL_NAME = "lbl8r_scvi_expr"
LBL8R_RAW_COUNT_MODEL_NAME = "lbl8r_raw_cnt"
# scanvi
SCANVI_BATCH_EQUALIZED_MODEL_NAME = "scanvi_batch_equal"
SCANVI_MODEL_NAME = "scanvi"
# e2e XGBoost model names
XGB_SCVI_EXPRESION_MODEL_NAME = "xgb_scvi_expr"
XGB_RAW_COUNT_MODEL_NAME = "xgb_raw_cnt"

# def setup_paths(model_params, data_path, config_path):
#     """
#     Create the paths for models and artifacts
#     """

#     # create the paths
#     model_path = Path(model_params["model_path"])
#     model_path.mkdir(exist_ok=True)

#     # create the data paths
#     data_path = Path(data_path)
#     data_path.mkdir(exist_ok=True)

#     # create the config paths
#     config_path = Path(config_path)
#     config_path.mkdir(exist_ok=True)

#     # create the paths for the artifacts

#     # create the paths for the models

#     # create the paths for the data

#     # create the paths for the config


# # def load_and_prep(data_path):
# #     """
# #     Load and prep the data
# #     """

# #     # prep

#     load_adata(data_path)


def create_artifacts(data: list, model: list, visualization_path, artifacts_path):
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

    ret_models = []
    # SCANVI E2E MODELS
    if model_name in (SCANVI_MODEL_NAME, SCANVI_BATCH_EQUALIZED_MODEL_NAME):
        # load teh scvi model, scanvi_model, (qnd query models?)
        vae_name = model_name.replace("lbl8r_scvi", "scvi")
        print(f"scanvi getting {(model_path/vae_name)}")
        vae, ad = get_trained_scvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=model_path,
            retrain=retrain,
            model_name=vae_name,
            **training_kwargs,
        )
        # add vae=vae to training_kwargs
        training_kwargs["vae"] = vae
        ret_models.append(vae)
        _get_model = get_trained_scanvi

    # other e2e models
    elif model_name in (LBL8R_RAW_COUNT_MODEL_NAME,LBL8R_SCVI_EXPRESION_MODEL_NAME) :
        # assume path for expression data variant already gets the right data
        _get_model = get_lbl8r

    elif model_name in (XGB_RAW_COUNT_MODEL_NAME, XGB_SCVI_EXPRESION_MODEL_NAME):
        # assume path for expression data variant already gets the right data
        _get_model = get_xgb

    # LBL8R Models
    elif model_name == SCVI_LATENT_MODEL_NAME:
        # need vae & scvi_emb model
        vae_name = model_name.replace("lbl8r_scvi", "scvi")
        print(f"getting {(model_path/vae_name)}")
        vae, ad = get_trained_scvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=model_path,
            retrain=retrain,
            model_name=vae_name,
            **training_kwargs,
        )
        # 1. get the latent representation
        ad = prep_latent_z_adata(ad, vae=vae, labels_key=labels_key)
        print(f"made latent ad")

        # 2. update the data with the latent representation
        data.update(ad)
        print(f"getting {(model_path/model_name)}")
        _get_model = get_lbl8r

        # # get_pca_lbl8r implicitly creates the latent ad... but 
        # _get_model = get_scvi_lbl8r

    elif model_name in (XGB_RAW_PC_MODEL_NAME, XGB_SCVI_EXPR_PC_MODEL_NAME):
        # assume full expr data already loaded
        # 1. get the pcs representation
        ad = prep_pcs_adata(data.adata, pca_key=PCA_KEY)
        # 2. update the data with the latent representation
        data.update(ad)
        
    elif model_name == SCVI_LATENT_MODEL_NAME:
        vae_name = SCVI_SUB_MODEL_NAME
        # get latent
        print(f"getting {(model_path/vae_name)}")
        vae, ad = get_trained_scvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=model_path,
            retrain=retrain,
            model_name=vae_name,
            **training_kwargs,
        )
        # 1. get the latent representation
        ad = prep_latent_z_adata(ad, vae=vae, labels_key=labels_key)
        print(f"made latent ad")
        # 2. update the data with the latent representation
        data.update(ad)

        _get_model = get_lbl8r


    # XGB LBL8RS
    elif model_name in  (XGB_RAW_PC_MODEL_NAME, XGB_SCVI_EXPR_PC_MODEL_NAME):
        # assume full expr data already loaded
        # 1. get the pcs representation
        ad = prep_pcs_adata(data.adata, pca_key=PCA_KEY)
        # 2. update the data with the latent representation
        data.update(ad)

        _get_model = get_xgb

    elif model_name == XGB_SCVI_LATENT_MODEL_NAME:
        vae_name = SCVI_SUB_MODEL_NAME
        # get latent
        print(f"getting {(model_path/vae_name)}")
        vae, ad = get_trained_scvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=model_path,
            retrain=retrain,
            model_name=vae_name,
            **training_kwargs,
        )
        # 1. get the latent representation
        ad = prep_latent_z_adata(ad, vae=vae, labels_key=labels_key)
        # 2. update the data with the latent representation
        data.update(ad)

        _get_model = get_xgb



    if model_name.startswith("scvi"):
        _get_model = get_trained_scvi

    elif model_name.startswith("scanvi"):
        # load the scvi model
        vae_name = SCVI_SUB_MODEL_NAME
        # get latent
        print(f"getting {(model_path/vae_name)}")
        vae, ad = get_trained_scvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=model_path,
            retrain=retrain,
            model_name=vae_name,
            **training_kwargs,
        )
        # add vae=vae to training_kwargs
        training_kwargs["vae"] = vae

        _get_model = get_trained_scanvi

        # add vae=vae to training_kwargs
        training_kwargs["scanvi_model"] = scanvi_model

        _get_model = get_query_scanvi

    # LBL8R models
    elif model_name.startswith("lbl8r"):
        if "pcs" in model_name:  # "lbl8r_cnt_pcs", "lbl8r_expr_pcs"
            # 1. get the pcs representation
            ad = prep_pcs_adata(data.adata, pca_key=PCA_KEY)
            # 2. update the data with the latent representation
            data.update(ad)
            _get_model = get_pca_lbl8r
        elif "scvi" in model_name:
            vae_name = model_name.replace("lbl8r_scvi", "scvi")
            print(f"getting {(model_path/vae_name)}")
            vae, ad = get_trained_scvi(
                data.adata,
                labels_key=labels_key,
                batch_key=None,
                model_path=model_path,
                retrain=retrain,
                model_name=vae_name,
                **training_kwargs,
            )
            # 1. get the latent representation
            ad = prep_latent_z_adata(ad, vae=vae, labels_key=labels_key)
            print(f"made latent ad")

            # 2. update the data with the latent representation
            data.update(ad)
            print(f"getting {(model_path/model_name)}")
            _get_model = get_lbl8r

    elif model_name.startswith("xgb"):
        if "pcs" in model_name:  # "lbl8r_cnt_pcs", "lbl8r_expr_pcs"
            # 1. get the pcs representation
            ad = prep_pcs_adata(data.adata, pca_key=PCA_KEY)
            # 2. update the data with the latent representation
            data.update(ad)

            _get_model = get_xgb
        elif "scvi" in model_name:
            # should have loaded embedding adata.  no prep nescessary
            #  _prep = [get_trained_scvi, prep_latent_z_adata]
            _get_model = get_xgb

    # E2E models
    elif model_name.startswith("raw_cnt") | model_name.startswith("scvi_expr_nb"):
        _get_model = get_lbl8r

    elif model_name.startswith("xgb_raw_cnt") | model_name.startswith("xgb_expr_nb"):
        _get_model = get_xgb

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

    # TODO: wrap the model in a Model class
    return model


def prep_latent_data(data: Adata, vae: SCVI, labels_key: str = "cell_type") -> Adata:
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    model :
        An classification model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    # 1. get the latent representation
    ad = prep_latent_z_adata(ad, vae=vae, labels_key=labels_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def prep_pc_data(data: Adata, pca_key=PCA_KEY) -> Adata:
    """
    Prep adata for pcs LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    pca_key : str
        Key for pca. Default is `X_pca`.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """
    # 1. get the pcs representation
    ad = prep_pcs_adata(data.adata, pca_key=pca_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def query_model(
    data: Adata,
    model: LBL8R | SCANVI | Booster,
    label_encoder: LabelEncoder | None = None,
    labels_key: str = "cell_type",
) -> Adata:
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

    if isinstance(model, SCANVI):
        # need to load / train query models

        # 1. get query_scvi (depricate?  needed for what?  latent conditioning?)

        # 2. get query_scanvi

        # 3. return model
        _query_model = query_scanvi
    # "transfer learning" query models which need to be trained
    elif model_name.startswith("query_scvi"):
        # load the scvi model
        vae_name = model_name.replace("lbl8r_scvi", "scvi")
        print(f"query_scvi getting {(model_path/vae_name)}")
        vae, ad = get_trained_scvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=model_path,
            retrain=retrain,
            model_name=vae_name,
            **training_kwargs,
        )
        # add vae=vae to training_kwargs
        training_kwargs["vae"] = vae

        _get_model = get_query_scvi

    elif model_name.startswith("query_scanvi"):
        # load the scvi model
        scanvi_name = model_name.replace("lbl8r_scvi", "scvi")
        print(f"query_scanvi getting {(model_path/vae_name)}")
        scanvi_model, ad = get_trained_scanvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=model_path,
            retrain=retrain,
            model_name=scanvi_name,
            **training_kwargs,
        )

    if isinstance(model, LBL8R):
        _query_model = query_lbl8r

    elif isinstance(model, Booster):
        _query_model = query_xgb

    else:
        raise ValueError(f"model {model} is not a valid model")

    ad = _query_model(data.adata, model, labels_key=labels_key)

    # update data with ad
    data.update(ad)

    return data
