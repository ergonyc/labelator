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
import scanpy as sc

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
    make_scvi_normalized_adata,
    make_latent_adata,
)

from .model._xgb import get_xgb, query_xgb
from .model.utils._data import Adata, transfer_pcs
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


def prep_model(
    data: Adata,
    model_name: str,
    model_path: Path | str,
    model: SCVI | SCANVI | LBL8R | None = None,
    labels_key: str = "cell_type",
    # batch_key: str | None = None, # TODO:  model_kwargs for controlling batch_key, etc..
    retrain: bool = False,
    **training_kwargs,
) -> Model:
    """
    Load a model from path or train and prep data

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
        print(f"scanvi getting {(model_path/model_name/SCVI_SUB_MODEL_NAME)}")
        vae, ad = get_trained_scvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=(model_path / model_name),
            retrain=retrain,
            model_name=SCVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        print(f"scanvi getting {(model_path/model_name/SCANVI_SUB_MODEL_NAME)}")
        model, ad = get_trained_scanvi(
            ad,
            vae,
            labels_key=labels_key,
            model_path=(model_path / model_name),
            retrain=retrain,
            model_name=SCANVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        # update data with ad
        data.update(ad)

        model = Model(model, model_path, model_name, vae, labels_key)
        return model

    elif model_name in (
        SCVI_LATENT_MODEL_NAME,
        XGB_SCVI_LATENT_MODEL_NAME,
        LBL8R_SCVI_EXPRESION_MODEL_NAME,
        SCVI_EXPR_PC_MODEL_NAME,
        XGB_SCVI_EXPRESION_MODEL_NAME,
        XGB_SCVI_EXPR_PC_MODEL_NAME,
    ):
        # 0. tag adata for artifact export
        data.set_out(model_name)

        # need vae
        print(f"getting scvi:{(model_path/SCVI_LATENT_MODEL_NAME/SCVI_SUB_MODEL_NAME)}")
        vae, ad = get_trained_scvi(
            data.adata,
            labels_key=labels_key,
            batch_key=None,
            model_path=(model_path / SCVI_LATENT_MODEL_NAME),
            retrain=retrain,
            model_name=SCVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        # SCVI LBL8R Model
        if model_name in (SCVI_LATENT_MODEL_NAME, XGB_SCVI_LATENT_MODEL_NAME):
            # 1. make the make_latent_adata
            ad = prep_latent_z_adata(ad, vae, labels_key=labels_key)

            if model_name == SCVI_LATENT_MODEL_NAME:
                print(f"getting {(model_path/model_name/LBL8R_SCVI_SUB_MODEL_NAME)}")
                model, ad = get_lbl8r(
                    ad,
                    labels_key=labels_key,
                    model_path=(model_path / model_name),
                    retrain=retrain,
                    model_name=LBL8R_SCVI_SUB_MODEL_NAME,
                    **training_kwargs,
                )
                # 2. update the data with the latent representation
                data.update(ad)

                model = Model(model, model_path, model_name, vae, labels_key)
                return model

            elif model_name == XGB_SCVI_LATENT_MODEL_NAME:
                print(f"getting {(model_path/model_name)}")
                model, ad = get_xgb(
                    ad,
                    labels_key=labels_key,
                    model_path=model_path,
                    retrain=retrain,
                    model_name=model_name,
                    **training_kwargs,
                )
                # 2. update the data with the latent representation
                data.update(ad)

                model = Model(model, model_path, model_name, vae, labels_key)
                return model

        elif model_name in (
            LBL8R_SCVI_EXPRESION_MODEL_NAME,
            XGB_SCVI_EXPRESION_MODEL_NAME,
            SCVI_EXPR_PC_MODEL_NAME,
            XGB_SCVI_EXPR_PC_MODEL_NAME,
        ):
            # 1. make scvi_expression data
            ad = make_scvi_normalized_adata(vae, ad)
            if model_name == LBL8R_SCVI_EXPRESION_MODEL_NAME:
                print(f"getting {(model_path/model_name)}")
                model, ad = get_lbl8r(
                    ad,
                    labels_key=labels_key,
                    model_path=model_path,
                    retrain=retrain,
                    model_name=model_name,
                    **training_kwargs,
                )
                # 2. update the data with the latent representation
                data.update(ad)
                model = Model(model, model_path, model_name, vae, labels_key)
                return model

            elif model_name == XGB_SCVI_EXPRESION_MODEL_NAME:
                print(f"getting {(model_path/model_name)}")
                model, ad = get_xgb(
                    ad,
                    labels_key=labels_key,
                    model_path=model_path,
                    retrain=retrain,
                    model_name=model_name,
                    **training_kwargs,
                )
                # 2. update the data with the latent representation
                data.update(ad)
                model = Model(model, model_path, model_name, vae, labels_key)
                return model

            elif model_name in (SCVI_EXPR_PC_MODEL_NAME, XGB_SCVI_EXPR_PC_MODEL_NAME):
                # 1. make the pcs representation
                ad = prep_pcs_adata(data.adata, pca_key=PCA_KEY)
                if model_name == SCVI_EXPR_PC_MODEL_NAME:
                    print(f"getting {(model_path/model_name)}")
                    model, ad = get_lbl8r(
                        ad,
                        labels_key=labels_key,
                        model_path=model_path,
                        retrain=retrain,
                        model_name=model_name,
                        **training_kwargs,
                    )
                    # 2. update the data with the latent representation
                    data.update(ad)
                    model = Model(model, model_path, model_name, vae, labels_key)
                    return model

                elif model_name == XGB_SCVI_EXPR_PC_MODEL_NAME:
                    print(f"getting {(model_path/model_name)}")
                    model, ad = get_xgb(
                        ad,
                        labels_key=labels_key,
                        model_path=model_path,
                        retrain=retrain,
                        model_name=model_name,
                        **training_kwargs,
                    )
                    # 2. update the data with the latent representation
                    data.update(ad)
                    model = Model(model, model_path, model_name, vae, labels_key)
                    return model

            else:
                raise ValueError(f"IMPOSSIBLE unknown model_name {model_name}")

    elif model_name in (RAW_PC_MODEL_NAME, XGB_RAW_PC_MODEL_NAME):
        # 1. get the pcs representation
        ad = prep_pcs_adata(data.adata, pca_key=PCA_KEY)
        # 2. update the data with the latent representation
        data.update(ad)
        if model_name == RAW_PC_MODEL_NAME:
            _get_model = get_lbl8r
        elif model_name == XGB_RAW_PC_MODEL_NAME:
            _get_model = get_xgb

    # other e2e models
    elif model_name in (LBL8R_RAW_COUNT_MODEL_NAME,):
        # assume path for expression data variant already gets the right data
        _get_model = get_lbl8r

    elif model_name in (XGB_RAW_COUNT_MODEL_NAME,):
        # assume path for expression data variant already gets the right data
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
    model = Model(model, model_path, model_name, None, labels_key)
    return model, data


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
    ad = prep_latent_z_adata(data.adata, vae=vae, labels_key=labels_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def prep_expr_data(data: Adata, vae: SCVI, ref_data: Adata | None = None) -> Adata:
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    vae :
        An scvi model used for normalization.
    ref_data : Adata
        "training" data used for transferring pcs

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    # 1. get the normalized adata
    ad = make_scvi_normalized_adata(vae, data.adata)

    # 2. update the data with pcs
    if ref_data is None:
        print("WARNING can't add PCs to normalized expression AnnData")
    else:
        ref = ref_data.adata
        if ref_data.adata.varm.get("PCs") is None:
            sc.pp.pca(ref)
        ref_data.update(ref)
        ad = transfer_pcs(ref, ad)

    # 3. update the data with the latent representation & pcs
    data.update(ad)
    return data


def prep_pc_data(data: Adata, pca_key=PCA_KEY, ref_data: Adata | None = None) -> Adata:
    """
    Prep adata for pcs LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    pca_key : str
        Key for pca. Default is `X_pca`.
    ref_data : Adata
        "training" data used for transferring pcs

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    ad = data.adata
    # 1. make sure we have pcs
    if data.adata.varm.get("PCs") is None:
        if ref_data is None:
            print("WARNING we don't have PCs in AnnData or reference to transfer")
        else:
            ref = ref_data.adata
            if ref.varm.get("PCs") is None:
                sc.pp.pca(ref)
                ref_data.update(ref)
            ad = transfer_pcs(ref, ad)

    # 1. get the pcs representation
    ad = prep_pcs_adata(ad, pca_key=pca_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def query_model(
    data: Adata,
    model: Model,
    retrain: bool = False,
    **kwargs,
) -> Adata:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model : LBL8R | SCANVI | Booster
        An classification model.
    insert_key: str
        Key for inserting predictions. Default is `pred`.
    *kwargs :
        Keyword arguments for `get_query_scvi` and `get_query_scanvi`.
    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    if isinstance(model.model, SCANVI):
        ad = query_scanvi(data.adata, model.model, insert_key=model.labels_key)

    # "transfer learning" query models which need to be trained

    # no prep needed to query these models.
    elif isinstance(model.model, LBL8R):
        ad = query_lbl8r(data.adata, model.model, labels_key=model.labels_key)

    elif isinstance(model.model, Booster):
        ad = query_xgb(data.adata, model.model, label_encoder=model.label_encoder)

    else:
        raise ValueError(f"model {model} is not a valid model")

    # update data with ad
    data.update(ad)

    return data


def prep_query_scanvi(
    data: Adata,
    model: Model,
    labels_key: str = "cell_type",
    retrain: bool = False,
) -> (Adata, Model):
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
    retrian : bool
        Retrain the model. Default is `False`.

    Returns
    -------
    Adata
        updated data
    Model
        Model with .q_vae, and .scanvi models added and querey_scanvi model as .model
    """

    ad = data.adata
    # 1. get query_scvi (depricate?  needed for what?  latent conditioning?)
    q_scvi, ad = get_query_scvi(
        ad,
        model.vae,
        labels_key=model.labels_key,
        model_path=(model.path / model.name),
        retrain=retrain,
        model_name=QUERY_SCVI_SUB_MODEL_NAME,
    )
    # 2. get query_scanvi
    q_scanvi, ad = get_query_scanvi(
        ad,
        model.model,
        labels_key=model.labels_key,
        model_path=(model.path / model.name),
        retrain=retrain,
        model_name=QUERY_SCANVI_SUB_MODEL_NAME,
    )
    # update data with ad
    data.update(ad)

    # 3. return model, pack all the models into the Model
    model.scanvi = model.model
    model.q_vae = q_scvi
    model.model = q_scanvi

    return data, model


# depricated...
def query_qscanvi(data: Adata, model: Model, insert_key: str = "label") -> Adata:
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

    ad = data.adata

    # 3. return model
    ad = query_scanvi(ad, model.model, insert_key=insert_key)
    # update data with ad
    data.update(ad)

    return data
