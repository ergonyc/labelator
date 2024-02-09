# interfrace for cli to lbl8 module
# %%
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
    add_latent_obsm,
)

from .model._xgb import get_xgb2, query_xgb, XGB
from .model.utils._data import Adata, transfer_pcs, prep_target_genes, merge_into_obs
from .model.utils._lazy_model import LazyModel, ModelSet
from .model.utils._plot import (
    plot_embedding,
    plot_predictions,
    plot_scvi_training,
    plot_scanvi_training,
    plot_lbl8r_training,
)
from .model.utils._artifact import (
    save_pcs,
    save_genes,
    load_genes,
    load_pcs,
    save_predictions,
    load_predictions,
)
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


def load_data(adata_path: str | Path, archive_path: str | Path) -> Adata:
    """
    Load training data.
    """

    if adata_path is None:
        return Adata(None)

    adata_path = Path(adata_path)
    data = Adata(adata_path)
    data.archive_path = archive_path
    return data


def load_trained_model(
    model_name: str,
    model_path: Path | str,
    labels_key: str = "cell_type",
):
    # lazily load models and pack into apropriate ModelSet

    # automatically loaded when initiationg ModelSet
    # pcs = load_pcs(model_path / model_name)
    # genes = load_genes(model_path / model_name)

    # SCANVI MODELS
    if model_name in (SCANVI_MODEL_NAME, SCANVI_BATCH_EQUALIZED_MODEL_NAME):
        vae_path = model_path / model_name / SCVI_SUB_MODEL_NAME
        vae = LazyModel(vae_path)
        # vae = SCVI.load(vae_path.as_posix())
        scanvi_path = model_path / model_name / SCANVI_SUB_MODEL_NAME
        scanvi_model = LazyModel(scanvi_path)
        models = {SCVI_SUB_MODEL_NAME: vae, SCANVI_SUB_MODEL_NAME: scanvi_model}
        model = ModelSet(models, (model_path / model_name), labels_key)
        model.batch_key = (
            "sample" if model_name == SCANVI_BATCH_EQUALIZED_MODEL_NAME else None
        )
        model.basis = SCANVI_LATENT_KEY

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
        model.basis = SCVI_LATENT_KEY

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
        model.basis = PCA_KEY
    else:
        raise ValueError(f"unknown model_name {model_name}")

    model.default = model_name
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
        print(f"prep_model 1: loading {(model_path/model_name)} model")
        model = load_trained_model(model_name, model_path, labels_key=labels_key)
        # if no traing data is loaded (just prepping for query) return placeholder data
        if data is None:
            print("no train_data passed")
            data = Adata(None)

        model.prepped = False
        return model, data

    elif data is not None:
        # train model.
        print(f"prep_model 1: training {(model_path/model_name)} model")
        data.labels_key = labels_key
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

    # 0. tag adata for artifact export
    data.set_output(model_name)
    # BUG. this coudl really be query data.  need to set to None and load adata from model? during prep?
    #  otherwise we need to load PCs artifacts so we can properly prep the query data.
    print(f"train_model: 0. ngenes = {ad.n_vars} ncells = {ad.n_obs}")
    # genes = ad.var_names.to_list()  # or should we leave as an Index?
    save_genes(ad, model_path / model_name)

    models = {}
    # SCANVI E2E MODELS
    if model_name in (SCANVI_MODEL_NAME, SCANVI_BATCH_EQUALIZED_MODEL_NAME):
        # pop the batch key...
        batch_key = (
            "sample" if model_name == SCANVI_BATCH_EQUALIZED_MODEL_NAME else None
        )

        # put the batch_key back in the training_kwargs
        training_kwargs["batch_key"] = batch_key

        # create a ground truth for SCANVI so we can clobber the labels_key for queries
        ad.obs["ground_truth"] = ad.obs[labels_key].to_list()

        # load teh scvi model, scanvi_model, (qnd query models?)
        print(f"scanvi getting 0 {(model_path/model_name/SCVI_SUB_MODEL_NAME)}")
        vae, ad = get_trained_scvi(
            ad,
            labels_key=labels_key,
            model_path=(model_path / model_name),
            retrain=retrain,
            model_name=SCVI_SUB_MODEL_NAME,
            **training_kwargs,
        )
        ad = add_latent_obsm(ad, vae)

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

        ad = add_latent_obsm(ad, model)
        # update data with ad
        data.labels_key = labels_key
        data.update(ad)
        save_pcs(ad, model_path / model_name)

        vae_path = model_path / model_name / SCVI_SUB_MODEL_NAME
        vae = LazyModel(vae_path, vae)
        # vae = SCVI.load(vae_path.as_posix())
        scanvi_path = model_path / model_name / SCANVI_SUB_MODEL_NAME
        scanvi_model = LazyModel(scanvi_path, model)
        models = {SCVI_SUB_MODEL_NAME: vae, SCANVI_SUB_MODEL_NAME: scanvi_model}

        model = ModelSet(models, (model_path / model_name), labels_key=labels_key)
        model.default = SCANVI_SUB_MODEL_NAME
        model.batch_key = batch_key
        model.basis = SCANVI_LATENT_KEY
        return model, data

    basis = PCA_KEY
    batch_key = None
    if model_name in (
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
            batch_key=batch_key,
            model_path=model_path,
            retrain=retrain,
            model_name=SCVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        vae_path = model_path / SCVI_SUB_MODEL_NAME
        vae = LazyModel(vae_path, vae)

        models = {SCVI_SUB_MODEL_NAME: vae}
        basis = SCVI_LATENT_KEY
        # SCVI LBL8R LazyModel
        if model_name in (SCVI_LATENT_MODEL_NAME, XGB_SCVI_LATENT_MODEL_NAME):
            # 1. make the make_latent_adata
            ad = prep_latent_z_adata(ad, vae.model, labels_key=labels_key)

        elif model_name in (
            SCVI_EXPRESION_MODEL_NAME,
            XGB_SCVI_EXPRESION_MODEL_NAME,
            SCVI_EXPR_PC_MODEL_NAME,
            XGB_SCVI_EXPR_PC_MODEL_NAME,
        ):
            # TODO:  load this data rather than compute it... saves 45s
            # 1. make scvi_expression data
            ad = make_scvi_normalized_adata(vae.model, ad)
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
        print(f"{model_name} doesn not need any data prep")

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
    data.labels_key = labels_key
    data.update(ad)
    save_pcs(ad, model_path / model_name)

    model = LazyModel(model_path / model_name, model)

    models.update({model_name: model})
    model = ModelSet(models, model_path / model_name, labels_key=labels_key)
    model.default = model_name

    # model.batch_key = None
    model.basis = basis

    return model, data


def prep_latent_data(
    data: Adata, vae: LazyModel, labels_key: str = "cell_type"
) -> Adata:
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    model : LazyModel
        An classification model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """
    if not vae.loaded:
        vae.load_model(data.adata)

    # 1. get the latent representation
    ad = prep_latent_z_adata(data.adata, vae=vae.model, labels_key=labels_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def prep_query_data(data: Adata, genes: list[str]) -> Adata:
    """
    Prep adata for query

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    genes : list[str]
        List of genes model expects (from training data)

    Returns
    -------
    Adata
        Adata with gene subsetted
    """
    print(f"prep_query_data: genes n= {len(genes)}")
    # ad = data.adata[:, genes].copy()

    # do we have ground truth labels?
    if data.labels_key is not None:
        data.ground_truth_key = data.labels_key
    else:
        # TODO: fix "ground_truth" logic
        print("should I set ground_truth_key to 'Unknown'?")
        # data.ground_truth_key = "Unknown"

    ad = prep_target_genes(data.adata, genes)
    data.update(ad)
    return data


def prep_expr_data(
    data: Adata, vae: LazyModel, ref_pcs: ndarray | None = None
) -> Adata:
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    vae : LazyModel
        An scvi model used for normalization.
    ref_pcs : ndarray
        "training" data used for transferring pcs

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """
    if not vae.loaded:
        vae.load_model(data.adata)

    # 1. get the normalized adata
    ad = make_scvi_normalized_adata(vae.model, data.adata)

    # 2. update the data with pcs
    if ref_pcs is not None:
        ad = transfer_pcs(ad, ref_ad=None, pcs=ref_pcs)
    else:
        ValueError("no PCs to transfer, can't make scvi normalized pcs")

    # 3. update the data with the latent representation & pcs
    data.update(ad)
    return data


def prep_pc_data(data: Adata, pca_key=PCA_KEY, ref_pcs: ndarray | None = None) -> Adata:
    """
    Prep adata for pcs LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    pca_key : str
        Key for pca. Default is `X_pca`.
    ref_pcs : ndarray
        "training" data used for transferring pcs

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    ad = data.adata
    # 1. make sure we have pcs
    if ref_pcs is not None:
        ad = transfer_pcs(ad, ref_ad=None, pcs=ref_pcs)
    elif pca_key in ad.obsm_keys():
        print(f"already have PCs. ?")
    elif "PCs" in ad.uns_keys():
        print(f"transfering PCs from .uns['PCs']")
        ref_pcs = ad.uns["PCs"].copy()
        ad = transfer_pcs(ad, ref_ad=None, pcs=ref_pcs)
    elif "PCs" in ad.varm_keys():
        print(f"transfering PCs from .varm['PCs']")
        ref_pcs = ad.varm["PCs"].copy()
        ad = transfer_pcs(ad, ref_ad=None, pcs=ref_pcs)
    else:
        ValueError("no PCs to transfer, can't make scvi normalized pcs")

    # 1. get the pcs representation
    ad = prep_pcs_adata(ad, pca_key=pca_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def query_model(
    data: Adata,
    model_set: ModelSet,
) -> Adata:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model_set : ModelSet
        A ModelSet of model parts for a classification model.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    model = model_set.model[model_set.default]

    if not model.loaded:
        model.load_model(data.adata)

    # TODO: update interface to query_scanvi, query_lbl8r, query_xgb so the predictiosn tables are returned
    #   do the merge_into_obs here, and also serialize the tables.

    ad = data.adata

    xgb_report = None
    if isinstance(model.model, SCANVI):
        # "transfer learning" query models which need to be trained
        predictions = query_scanvi(ad, model.model)

        # fix the labels_key with "ground_truth"
        ad.obs[model_set.labels_key] = ad.obs["ground_truth"].to_list()
    # no prep needed to query these models.
    elif isinstance(model.model, LBL8R):
        predictions = query_lbl8r(ad, model.model)
    elif isinstance(model.model, XGB):
        predictions, xgb_report = query_xgb(ad, model.model)
        # TODO: do something with the report...
    else:
        raise ValueError(f"model {model.model} is not a valid model")

    ad = merge_into_obs(ad, predictions)

    # update data with ad
    data.update(ad)
    # TODO: load the predictions into the model_set.. need to keep "train" validation and "query" predictions separate
    # TODO: naming convention for saving datat
    model_set.predictions[data.name] = predictions
    model_set.report[data.name] = xgb_report

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
    batch_key = model_set.batch_key
    trained_genes = model_set.genes

    # the query data doesn't automatically know the batch_key.  ADD IT

    # does this work?  what if we don't have all the trained genes, will it fill them wiht zeros?
    ad = ad[:, trained_genes].copy()

    # create a ground truth for SCANVI so we can clobber the labels_key for queries
    ad.obs["ground_truth"] = ad.obs[labels_key].to_list()
    ad.obs[labels_key] = "Unknown"

    # 1. get query_scvi (depricate?  needed for what?  latent conditioning?)
    q_scvi, ad = get_query_scvi(
        ad,
        model_set.model[SCVI_SUB_MODEL_NAME].model,
        labels_key=labels_key,
        batch_key=batch_key,
        model_path=model_set.path,
        retrain=retrain,
        model_name=f"{QUERY_SCVI_SUB_MODEL_NAME}_{data.name.rstrip('.h5ad')}",
    )

    # add latent representation to ad for embedding plots..
    ad = add_latent_obsm(ad, q_scvi)

    # 2. get query_scanvi
    q_scanvi, ad = get_query_scanvi(
        ad,
        model_set.model[SCANVI_SUB_MODEL_NAME].model,
        labels_key=labels_key,
        batch_key=batch_key,
        model_path=model_set.path,
        retrain=retrain,
        model_name=f"{QUERY_SCANVI_SUB_MODEL_NAME}_{data.name.rstrip('.h5ad')}",
    )

    ad = add_latent_obsm(ad, q_scanvi)
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
    model_set.add_model(models)
    model_set.default = QUERY_SCANVI_SUB_MODEL_NAME
    model_set.basis = SCANVI_LATENT_KEY
    return data, model_set


def prep_query_model(
    query_data: Adata,
    model_set: ModelSet,
    model_name: str,
    labels_key: str = "cell_type",
    retrain: bool = False,
) -> (ModelSet, Adata):
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
    ModelSet
        A ModelSet with the query models added
    Adata
        Annotated data matrix with latent variables as X

    """

    # 0. tag adata for artifact export
    query_data.set_output(model_name)

    # set the labels_key
    query_data.labels_key = labels_key
    genes = model_set.genes
    # NOTE: scanvi / scvi scarches mixin prep_query_data automatically handles this...
    query_data = prep_query_data(query_data, genes)

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
            query_data, model_set.model[SCVI_SUB_MODEL_NAME], ref_pcs=model_set.pcs
        )
        if model_name in (
            SCVI_EXPR_PC_MODEL_NAME,
            XGB_SCVI_EXPR_PC_MODEL_NAME,
        ):
            query_data = prep_pc_data(query_data, ref_pcs=model_set.pcs)

    elif model_name in (
        SCVI_LATENT_MODEL_NAME,
        XGB_SCVI_LATENT_MODEL_NAME,
    ):
        # SCVI embedding models
        query_data = prep_latent_data(
            query_data,
            model_set.model[SCVI_SUB_MODEL_NAME],
            labels_key=labels_key,
        )

    elif model_name in (
        RAW_PC_MODEL_NAME,
        XGB_RAW_PC_MODEL_NAME,
    ):
        # PCS models
        query_data = prep_pc_data(query_data, ref_pcs=model_set.pcs)

    elif model_name in (
        SCANVI_BATCH_EQUALIZED_MODEL_NAME,
        SCANVI_MODEL_NAME,
    ):
        # SCANVI models actually need to be fit (transfer learning...)
        query_data, model_set = prep_query_scanvi(
            query_data,
            model_set,
            retrain=retrain,
        )

    return model_set, query_data


def archive_artifacts(
    train_data: Adata, query_data: Adata, model_set: ModelSet, path: Path
):
    """
    Archive the artifacts
        - plots
        - adata

    """

    if train_data.adata is not None:  # just in case we are only "querying" or "getting"
        print(f"archive training plots and data: {'📈 '*25}")
        archive_plots(train_data, model_set, "train", fig_path=(path / "figs"))
        print(f"archive train output adata: {'💾 '*25}")
        archive_data(train_data)

    if query_data.adata is not None:
        print(f"archive query plots and data: {'📊 '*25}")
        archive_plots(query_data, model_set, "query", fig_path=(path / "figs"))
        print(f"archive query output adata: {'💾 '*25}")
        archive_data(query_data)


def archive_plots(
    data: Adata,
    model_set: ModelSet,
    train_or_query: str,
    fig_path: Path | str | None = None,
):
    """
    Archive the plots
    """
    ad = data.adata
    labels_key = data.labels_key
    ground_truth_key = data.ground_truth_key
    main_model = model_set.model[model_set.default]

    ## training args
    figs = []
    if main_model.type == "scanvi":
        # load model
        fig_dir = fig_path / main_model.name
        if train_or_query == "train":
            file_nm = f"{train_or_query.upper()}_{SCVI_SUB_MODEL_NAME}_{data.name.rstrip('.h5ad')}"
            fig_kwargs = dict(fig_dir=fig_dir, fig_nm=file_nm, show=False)
            fg = plot_scvi_training(
                model_set.model[SCVI_SUB_MODEL_NAME].model.history, **fig_kwargs
            )
            figs.extend(fg)

            file_nm = f"{train_or_query.upper()}_{SCANVI_SUB_MODEL_NAME}_{data.name.rstrip('.h5ad')}"
            fig_kwargs = dict(fig_dir=fig_dir, fig_nm=file_nm, show=False)
            fg = plot_scanvi_training(
                model_set.model[SCANVI_SUB_MODEL_NAME].model.history, **fig_kwargs
            )
            figs.extend(fg)
        else:
            file_nm = f"{train_or_query.upper()}_{QUERY_SCVI_SUB_MODEL_NAME}_{data.name.rstrip('.h5ad')}"
            fig_kwargs = dict(fig_dir=fig_dir, fig_nm=file_nm, show=False)
            fg = plot_scvi_training(
                model_set.model[QUERY_SCVI_SUB_MODEL_NAME].model.history, **fig_kwargs
            )
            figs.extend(fg)

            file_nm = f"{train_or_query.upper()}_{QUERY_SCANVI_SUB_MODEL_NAME}_{data.name.rstrip('.h5ad')}"
            fig_kwargs = dict(fig_dir=fig_dir, fig_nm=file_nm, show=False)
            fg = plot_scanvi_training(
                model_set.model[QUERY_SCANVI_SUB_MODEL_NAME].model.history, **fig_kwargs
            )
            figs.extend(fg)

    elif main_model.type == "lbl8r":
        fig_dir = fig_path / main_model.name
        file_nm = (
            f"{train_or_query.upper()}_{main_model.name}_{data.name.rstrip('.h5ad')}"
        )
        fig_kwargs = dict(fig_dir=fig_dir, fig_nm=file_nm, show=False)
        fg = plot_lbl8r_training(main_model.model.history, **fig_kwargs)
        figs.extend(fg)

    elif main_model.type == "xgb":
        print("plotting for XGBoost training not available")

    else:
        raise ValueError(f"model_type must be one of: 'scanvi', 'lbl8r', 'xgb'")

    fig_nm = f"{train_or_query.upper()}_{main_model.name}_{data.name.rstrip('.h5ad')}"
    fig_dir = fig_path / main_model.name

    # predictions
    title_str = fig_nm.replace("_", "-")
    if ground_truth_key is not None:
        fg = plot_predictions(
            ad,
            pred_key="pred",
            cell_type_key=labels_key,
            model_name=main_model.name,
            fig_nm=fig_nm,
            fig_dir=fig_dir,
            show=False,
        )
        figs.append(fg)
    else:
        print("no ground truth labels to plot")

    # embeddings

    # PLOT embeddings ###############################################################
    fg = plot_embedding(
        ad,
        fig_nm,
        fig_dir=fig_dir,
        basis=model_set.basis,
        color=[labels_key, "pred", "batch"],
        show=False,
    )
    figs.append(fg)

    ## save plots..
    for fig in figs:
        fig.savefig()


def archive_data(
    data: Adata,
    path: Path | None = None,
):
    """
    Archive the data

    exports the output data as anndata.
    other artifacts were generated along the way:
        - genes
        - pcs
        - predictions
        - plots
    """

    data.export(path)


# %%
