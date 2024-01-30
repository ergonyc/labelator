# interfrace for cli to lbl8 module

from anndata import AnnData
from scvi.model import SCVI, SCANVI
from pathlib import Path

# from xgboost import Booster
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import pickle
from numpy import ndarray

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

from .model._xgb import get_xgb2, query_xgb, load_xgboost, XGB
from .model.utils._data import Adata, transfer_pcs
from .model.utils._lazy_model import LazyModel, ModelSet
from .model.utils._plot import make_plots
from .model.utils._pcs import save_pcs, load_pcs, extract_pcs
from ._constants import *

MODEL = SCVI | SCANVI | LBL8R | XGB

PRED_KEY = "label"


INSERT_KEY = "pred"


# TODO:  make the code this so we can save tables instead of always writing anndata
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

    if adata_path is None:
        return Adata(None)

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


def load_trained_model(
    model_name: str,
    model_path: Path | str,
    labels_key: str = "cell_type",
):
    # lazily load models and pack into apropriate ModelSet

    # SCANVI MODELS
    if model_name in (SCANVI_MODEL_NAME, SCANVI_BATCH_EQUALIZED_MODEL_NAME):
        vae_path = model_path / model_name / SCVI_SUB_MODEL_NAME
        vae = LazyModel(vae_path)
        # vae = SCVI.load(vae_path.as_posix())
        scanvi_path = model_path / model_name / SCANVI_SUB_MODEL_NAME
        scanvi_model = LazyModel(scanvi_path)
        models = {SCVI_SUB_MODEL_NAME: vae, SCANVI_SUB_MODEL_NAME: scanvi_model}
        model = ModelSet(models, (model_path / model_name), labels_key)

    # scvi REPR models
    elif model_name in (
        SCVI_LATENT_MODEL_NAME,
        SCVI_EXPR_PC_MODEL_NAME,
        SCVI_EXPRESION_MODEL_NAME,
        XGB_SCVI_LATENT_MODEL_NAME,
        XGB_SCVI_EXPRESION_MODEL_NAME,
        XGB_SCVI_EXPR_PC_MODEL_NAME,
    ):
        # need vae
        vae_path = model_path / SCVI_SUB_MODEL_NAME
        vae = LazyModel(vae_path)
        model_path = model_path / model_name
        model = LazyModel(model_path)
        models = {model_name: model, SCVI_SUB_MODEL_NAME: vae}
        model = ModelSet(models, model_path, labels_key)

    # CNT models
    elif model_name in (
        RAW_PC_MODEL_NAME,
        RAW_COUNT_MODEL_NAME,
        XGB_RAW_PC_MODEL_NAME,
        XGB_RAW_COUNT_MODEL_NAME,
    ):
        model_path = model_path / model_name
        model = LazyModel(model_path)
        models = {model_name: model}
        model = ModelSet(models, model_path, labels_key)

    else:
        raise ValueError(f"unknown model_name {model_name}")

    return model


def prep_model(
    data: Adata,
    model_name: str,
    model_path: Path | str,
    labels_key: str = "cell_type",
    retrain: bool = False,
    **training_kwargs,
) -> tuple[ModelSet, Adata]:
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
    print(f"prep_model0: getting {(model_path/model_name)} model")

    if model_path.exists() and not retrain and data is None:
        # lazy load model
        model = load_trained_model(model_name, model_path, labels_key=labels_key)
        # if no traing data is loaded (just prepping for query) return placeholder data
        if data is None:
            data = Adata(None)

        model.prepped = False
        return model, data

    elif data is not None:
        # train model.
        model, data = train_model(
            data,
            model_name,
            model_path,
            labels_key=labels_key,
            retrain=retrain,
            **training_kwargs,
        )

        model.prepped = True
        return model, data

    else:
        raise ValueError(
            f"No trained{model_name} at {model_path}. Need training `data` to train model."
        )

        # we don't have trainign data so can only load a model.


def train_model(
    data: Adata,
    model_name: str,
    model_path: Path | str,
    labels_key: str = "cell_type",
    # batch_key: str | None = None, # TODO:  model_kwargs for controlling batch_key, etc..
    retrain: bool = False,
    **training_kwargs,
) -> ModelSet:
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
    ad = data.adata
    batch_key = training_kwargs.pop("batch_key", None)

    # 0. tag adata for artifact export
    data.set_output(model_name)
    # BUG. this coudl really be query data.  need to set to None and load adata from model? during prep?
    #  otherwise we need to load PCs artifacts so we can properly prep the query data.

    models = {}
    # SCANVI E2E MODELS
    if model_name in (SCANVI_MODEL_NAME, SCANVI_BATCH_EQUALIZED_MODEL_NAME):
        batch_key = (
            "sample" if model_name == SCANVI_BATCH_EQUALIZED_MODEL_NAME else None
        )

        # create a ground truth for SCANVI so we can clobber the labels_key for queries
        ad.obs["ground_truth"] = ad.obs[labels_key].to_list()

        # load teh scvi model, scanvi_model, (qnd query models?)
        print(f"scanvi getting 0 {(model_path/model_name/SCVI_SUB_MODEL_NAME)}")
        vae, ad = get_trained_scvi(
            ad,
            labels_key=labels_key,
            batch_key=batch_key,
            model_path=(model_path / model_name),
            retrain=retrain,
            model_name=SCVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        print(f"scanvi getting 1 {(model_path/model_name/SCANVI_SUB_MODEL_NAME)}")
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
        save_pcs(ad, model_path / model_name)

        vae_path = model_path / model_name / SCVI_SUB_MODEL_NAME
        vae = LazyModel(vae_path, vae)
        # vae = SCVI.load(vae_path.as_posix())
        scanvi_path = model_path / model_name / SCANVI_SUB_MODEL_NAME
        scanvi_model = LazyModel(scanvi_path, scanvi_model)
        models = {SCVI_SUB_MODEL_NAME: vae, SCANVI_SUB_MODEL_NAME: scanvi_model}

        model = ModelSet(models, (model_path / model_name), labels_key)
        return model, data

    elif model_name in (
        SCVI_LATENT_MODEL_NAME,
        XGB_SCVI_LATENT_MODEL_NAME,
        SCVI_EXPRESION_MODEL_NAME,
        SCVI_EXPR_PC_MODEL_NAME,
        XGB_SCVI_EXPRESION_MODEL_NAME,
        XGB_SCVI_EXPR_PC_MODEL_NAME,
    ):
        # need vae
        print(f"getting scvi: 1 {(model_path/SCVI_SUB_MODEL_NAME)}")
        vae, ad = get_trained_scvi(
            ad,
            labels_key=labels_key,
            batch_key=None,
            model_path=model_path,
            retrain=retrain,
            model_name=SCVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        vae_path = model_path / SCVI_SUB_MODEL_NAME
        vae = LazyModel(vae_path, vae)

        models = {SCVI_SUB_MODEL_NAME: vae}

        # SCVI LBL8R LazyModel
        if model_name in (SCVI_LATENT_MODEL_NAME, XGB_SCVI_LATENT_MODEL_NAME):
            # 1. make the make_latent_adata
            ad = prep_latent_z_adata(ad, vae, labels_key=labels_key)

        elif model_name in (
            SCVI_EXPRESION_MODEL_NAME,
            XGB_SCVI_EXPRESION_MODEL_NAME,
            SCVI_EXPR_PC_MODEL_NAME,
            XGB_SCVI_EXPR_PC_MODEL_NAME,
        ):
            # TODO:  load this data rather than compute it... saves 45s
            # 1. make scvi_expression data
            ad = make_scvi_normalized_adata(vae, ad)
            # 2. update the data with pcs
            sc.pp.pca(ad)
            # 3. update the data with the latent representation
            # ad.obsm["X_scVI"] = vae.get_latent_representation(ad)

            if model_name in (SCVI_EXPR_PC_MODEL_NAME, XGB_SCVI_EXPR_PC_MODEL_NAME):
                # 1. make the pcs representation
                ad = prep_pcs_adata(ad, pca_key=PCA_KEY)

    elif model_name in (RAW_PC_MODEL_NAME, XGB_RAW_PC_MODEL_NAME):
        print(f"getting raw pcs {(model_path/model_name)}")

        # 1. get the pcs representation
        ad = prep_pcs_adata(ad, pca_key=PCA_KEY)
        # 2. update the data with the latent representation
        data.update(ad)

    else:
        # RAW model no update to adata

        raise ValueError(f"unknown model_name {model_name}")

    if model_name.endswith("_xgb"):
        get_model = get_xgb2
    else:
        get_model = get_lbl8r

    model, ad = get_model(
        ad,
        labels_key=labels_key,
        model_path=model_path,
        retrain=retrain,
        model_name=model_name,
        **training_kwargs,
    )
    # 2. update the data with the latent representation
    data.update(ad)
    save_pcs(ad, model_path / model_name)

    model = LazyModel(model_path / model_name, model)

    models.update({model_name: model})
    model = ModelSet(models, model_path / model_name, labels_key)

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


def prep_expr_data(
    data: Adata, vae: SCVI, ref_data: Adata | ndarray | None = None
) -> Adata:
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
    if isinstance(ref_data, Adata):
        ref = ref_data.adata
        ad = transfer_pcs(ad, ref_ad=ref, pcs=None)
    elif isinstance(ref_data, ndarray):
        ad = transfer_pcs(ad, ref_ad=None, pcs=ref_data)
    else:
        ValueError("no PCs to transfer, can't make scvi normalized pcs")

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
    if isinstance(ref_data, Adata):
        ref = ref_data.adata
        ad = transfer_pcs(ad, ref_ad=ref, pcs=None)
    elif isinstance(ref_data, ndarray):
        ad = transfer_pcs(ad, ref_ad=None, pcs=ref_data)
    else:
        ValueError("no PCs to transfer, can't make scvi normalized pcs")

    # 1. get the pcs representation
    ad = prep_pcs_adata(ad, pca_key=pca_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def query_model(
    data: Adata,
    model: LazyModel,
) -> Adata:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model : LBL8R | SCANVI | Booster
        An classification model.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    if isinstance(model.model, SCANVI):
        # "transfer learning" query models which need to be trained
        ad = query_scanvi(data.adata, model.model)

        # fix the labels_key with "ground_truth"
        ad.obs[model.labels_key] = ad.obs["ground_truth"].to_list()

    # no prep needed to query these models.
    elif isinstance(model.model, LBL8R):
        ad = query_lbl8r(data.adata, model.model)
    elif isinstance(model.model, XGB):
        ad = query_xgb(data.adata, model.model)
        # TODO: do something with the report...
    else:
        raise ValueError(f"model {model} is not a valid model")

    # update data with ad
    data.update(ad)

    return data


def prep_query_scanvi(
    data: Adata,
    model_set: ModelSet,
    retrain: bool = False,
) -> (Adata, ModelSet):
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    model_set :
        An ModelSet of classification models.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    retrian : bool
        Retrain the model. Default is `False`.

    Returns
    -------
    Adata
        updated data
    LazyModel
        LazyModel with .q_vae, and .scanvi models added and querey_scanvi model as .model
    """

    ad = data.adata

    labels_key = model_set.labels_key

    # create a ground truth for SCANVI so we can clobber the labels_key for queries
    ad.obs["ground_truth"] = ad.obs[labels_key].to_list()
    ad.obs[labels_key] = "Unknown"

    # 1. get query_scvi (depricate?  needed for what?  latent conditioning?)
    q_scvi, ad = get_query_scvi(
        ad,
        model_set[SCVI_SUB_MODEL_NAME],
        labels_key=labels_key,
        model_path=model_set.path,
        retrain=retrain,
        model_name=QUERY_SCVI_SUB_MODEL_NAME,
    )
    # 2. get query_scanvi
    q_scanvi, ad = get_query_scanvi(
        ad,
        model_set[SCANVI_SUB_MODEL_NAME],
        labels_key=labels_key,
        model_path=model_set.path,
        retrain=retrain,
        model_name=QUERY_SCANVI_SUB_MODEL_NAME,
    )

    data = prep_pc_data(data, ref_data=model_set.pcs)

    # update data with ad
    data.update(ad)

    # 3. return model, pack all the models into the LazyModel
    qscvi_path = model_set.path / SCVI_SUB_MODEL_NAME
    q_scvi = LazyModel(qscvi_path, q_scvi)
    # vae = SCVI.load(vae_path.as_posix())
    qscanvi_path = model_set.path / SCANVI_SUB_MODEL_NAME
    q_scanvi = LazyModel(qscanvi_path, q_scanvi)
    models = {QUERY_SCVI_SUB_MODEL_NAME: q_scvi, QUERY_SCANVI_SUB_MODEL_NAME: q_scanvi}

    # 4 pack into the model set
    model_set.update(models)
    return data, model_set


# BUG. this coudl really be query data.  need to set to None and load adata from model? during prep?
#  otherwise we need to load PCs artifacts so we can properly prep the query data.
def prep_query_model(
    query_data: Adata,
    model_set: ModelSet,
    model_name: str,
    ref_data: Adata | None = None,
    labels_key: str = "cell_type",
    retrain: bool = False,
):
    """
    Prep Adata and ModelSet for inference

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    model :
        A classification ModelSet.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    retrian : bool
        Retrain the model. Default is `False`.

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    # 0. tag adata for artifact export
    query_data.set_output(model_name)

    # if we didn't load the training data we need to load the PCs from the model
    # might just want to always do this and not pass in ref_data.
    if ref_data is None:
        ref_data = model_set.pcs

    # 1. prep query data (normalize / get latents / transfer PCs (if normalized) )
    if model_name in (
        SCVI_EXPRESION_MODEL_NAME,
        XGB_SCVI_EXPRESION_MODEL_NAME,
        SCVI_EXPR_PC_MODEL_NAME,
        XGB_SCVI_EXPR_PC_MODEL_NAME,
    ):
        # add latent representation to query_data?
        # ad.obsm["X_scVI"] = model.vae.get_latent_representation(query_data.data)
        # query_data.update(ad)
        # TODO:  in order to get the pcs we need the reference vectors (PCs) from training data
        # SCVI expression models
        pcs = model_set.pcs
        query_data = prep_expr_data(
            query_data, model_set.model[SCVI_SUB_MODEL_NAME], ref_data=ref_data
        )
        if model_name in (
            SCVI_EXPR_PC_MODEL_NAME,
            XGB_SCVI_EXPR_PC_MODEL_NAME,
        ):
            query_data = prep_pc_data(query_data, ref_data=ref_data)

    elif model_name in (
        SCVI_LATENT_MODEL_NAME,
        XGB_SCVI_LATENT_MODEL_NAME,
    ):
        # SCVI embedding models
        query_data = prep_latent_data(
            query_data, model_set.model[SCVI_SUB_MODEL_NAME], labels_key=labels_key
        )

    elif model_name in (
        RAW_PC_MODEL_NAME,
        XGB_RAW_PC_MODEL_NAME,
    ):
        # PCS models
        query_data = prep_pc_data(query_data, ref_data=ref_data)

    elif model_name in (
        SCANVI_BATCH_EQUALIZED_MODEL_NAME,
        SCANVI_MODEL_NAME,
    ):
        # SCANVI models actually need to be fit (transfer learning...)
        query_data, model_set = prep_query_scanvi(
            query_data,
            model_set,
            labels_key=labels_key,
            retrain=retrain,
        )

    return query_data, model_set


def archive_plots(
    data: Adata,
    model: LazyModel,
    train_or_query: str,
    labels_key: str,
    path: Path | str | None = None,
):
    """
    Archive the plots
    """
    figs = make_plots(data, model, train_or_query, labels_key=labels_key, path=path)

    ## save plots..
    for fig in figs:
        fig.savefig()


def archive_data(
    data: Adata,
    path: Path,
):
    """
    Archive the data
    """

    data.export(path)
