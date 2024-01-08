from anndata import AnnData
import numpy as np
from scvi.model import SCVI, SCANVI
from pathlib import Path

from ._models import LBL8R, scviLBL8R
from .utils import (
    mde,
    make_latent_adata,
    add_predictions_to_adata,
    plot_predictions,
    plot_embedding,
    add_scanvi_predictions,
    export_ouput_adata,
    plot_scvi_training,
    plot_scanvi_training,
    plot_lbl8r_training,
    make_pc_loading_adata,
)

SCVI_LATENT_KEY = "X_scVI"
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTIONS_KEY = "C_scANVI"


# TODO: add save and load flags so we can use the functions and NOT overwrite on accident
def get_trained_scvi(
    adata: AnnData,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "scvi",
    plot_training: bool = False,
    **training_kwargs,
) -> (SCVI, AnnData):
    """
    Get scVI model and add latent representation to adata

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    batch_key : str
        Key for batch labels. Default is `None`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCVI`.
    plot_training : bool
        Whether to plot training. Default is `False`.
    **training_kwargs : dict
        Additional arguments to pass to `scvi.model.SCVI.train`.

    Returns
    -------
    SCVI
        scVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    scvi_epochs = 200
    batch_size = 512

    layer = None
    categorical_covariate_keys = (
        None  # ['sample', 'batch'] Currently limited to single categorical...
    )
    continuous_covariate_keys = (
        None  # noise = ['doublet_score', 'percent.mt', 'percent.rb'] # aka "noise"
    )
    size_factor_key = None  # library size

    SCVI.setup_anndata(
        adata,
        batch_key=batch_key,  # using this to prevent issues with categorical covariates
        layer=layer,
        labels_key=labels_key,
        categorical_covariate_keys=categorical_covariate_keys,
        continuous_covariate_keys=continuous_covariate_keys,
        size_factor_key=size_factor_key,
    )  # X contains raw counts

    scvi_path = model_path / model_name
    if scvi_path.exists() and not retrain:
        vae = SCVI.load(scvi_path.as_posix(), adata.copy())

    else:
        training_kwargs = {}
        vae = SCVI(
            adata,
            n_layers=2,
            encode_covariates=False,  # True
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none",
        )  # .cuda()
        vae.train(
            max_epochs=scvi_epochs,
            train_size=0.85,
            batch_size=batch_size,
            early_stopping=True,
            **training_kwargs,
        )

    adata.obsm[SCVI_LATENT_KEY] = vae.get_latent_representation()

    if retrain or not scvi_path.exists():
        # save the reference model
        vae.save(scvi_path, overwrite=True)

    if plot_training:
        plot_scvi_training(vae.history)
    return vae, adata


def get_trained_scanvi(
    adata: AnnData,
    vae: SCVI | None = None,
    labels_key: str = "cell_type",
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "scanvi",
    plot_training: bool = False,
    **training_kwargs,
) -> (SCANVI, AnnData):
    """
    Get scANVI model and add latent representation to adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    scanvi_model : SCANVI
        An scANVI model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCANVI`.
    plot_training : bool
        Whether to plot training. Default is `False`.
    plot_preds : bool
        Whether to plot predictions. Default is `True`.
    **training_kwargs : dict
        Additional arguments to pass to `scvi.model.SCANVI.train`.

    Returns
    -------
    SCANVI
        scANVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    scvi_epochs = 200
    batch_size = 512

    # n_labels = len(adata.obs[labels_key].cat.categories)
    batch_key = training_kwargs.pop("batch_key", None)

    if (
        vae is None
    ):  # get the scVI model.  Note the desired batch_key needs to be passed
        # BUG: this will overwrite the SCVI model... need to fix
        scvi_model_name = "SCVI" + model_name
        vae, adata = get_trained_scvi(
            adata,
            labels_key=labels_key,
            batch_key=batch_key,
            model_path=model_path,
            retrain=False,
            model_name=scvi_model_name,
            plot_training=True,
            **training_kwargs,
        )

    scanvi_path = model_path / model_name
    if scanvi_path.exists() and not retrain:
        scanvi_model = SCANVI.load(scanvi_path.as_posix(), adata)
    else:
        scanvi_model = SCANVI.from_scvi_model(vae, "Unknown", labels_key=labels_key)
        scanvi_model.train(
            max_epochs=scvi_epochs,
            train_size=0.85,
            batch_size=batch_size,
            early_stopping=True,
            **training_kwargs,
        )
    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
    adata = add_scanvi_predictions(
        adata, scanvi_model, insert_key=SCANVI_PREDICTIONS_KEY
    )

    if retrain or not scanvi_path.exists():
        # save the reference model
        scanvi_model.save(scanvi_path, overwrite=True)

    if plot_training:
        plot_scanvi_training(scanvi_model.history)

    return scanvi_model, adata


def query_scvi(
    adata: AnnData,
    vae: SCVI | None = None,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "query_scvi",
    plot_training: bool = False,
    **training_kwargs,
) -> (SCVI, AnnData):
    """
    Get scVI model via `scarches` surgery and add latent representation to adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    vae : SCVI
        An scVI model. None by default which calls get_train_scvi.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    batch_key : str
        Key for batch labels. Default is `None`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCVI_query`. but can change to use query_scvi for lbl8r
    plot_training : bool
        Whether to plot training. Default is `False`.
    **training_kwargs : dict
        Additional arguments to pass to `scvi.model.SCVI.train`.

    Returns
    -------
    SCVI
        scVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    surgery_epochs = 150

    qscvi_path = model_path / model_name

    batch_key = training_kwargs.pop("batch_key", None)
    if (
        vae is None
    ):  # get the scVI model.  Note the desired batch_key needs to be passed
        vae, adata = get_trained_scvi(
            adata,
            labels_key=labels_key,
            batch_key=batch_key,
            model_path=model_path,
            retrain=False,
            plot_training=True,
            **training_kwargs,
        )

    SCVI.prepare_query_anndata(adata, vae)
    if qscvi_path.exists() and not retrain:
        scvi_query = SCVI.load(qscvi_path.as_posix(), adata)
    else:
        scvi_query = SCVI.load_query_data(adata, vae)
        scvi_query.train(
            max_epochs=surgery_epochs,
            train_size=0.85,
            early_stopping=True,
            plan_kwargs=dict(weight_decay=0.0),
        )

    adata.obsm[SCVI_LATENT_KEY] = scvi_query.get_latent_representation(adata)
    if retrain or not qscvi_path.exists():
        # save the reference model
        scvi_query.save(qscvi_path, overwrite=True)

    if plot_training:
        plot_scvi_training(scvi_query.history)

    return scvi_query, adata


def query_scanvi(
    adata: AnnData,
    scanvi_model: SCANVI,
    labels_key: str = "cell_type",
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "query_scanvi",
    plot_training: bool = False,
    **training_kwargs,
) -> (SCANVI, AnnData):
    """
    Get scANVI model via `scarches` surgery and add latent representation to adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    scanvi_model : SCANVI
        A scANVI model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCANVI_query`.
    plot_training : bool
        Whether to plot training. Default is `False`.
    plot_preds : bool
        Whether to plot predictions. Default is `True`.
    **training_kwargs : dict
        Additional arguments to pass to `scvi.model.SCANVI.train`.

    Returns
    -------
    SCANVI
        scANVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    surgery_epochs = 150

    qscanvi_path = model_path / model_name
    SCANVI.prepare_query_anndata(adata, scanvi_model)

    if qscanvi_path.exists() and not retrain:
        scanvi_query = SCANVI.load(qscanvi_path.as_posix(), adata)
    else:
        scanvi_query = SCANVI.load_query_data(adata, scanvi_model)
        scanvi_query.train(
            max_epochs=surgery_epochs,
            train_size=0.85,
            early_stopping=True,
            plan_kwargs=dict(weight_decay=0.0),
        )

    adata.obsm[SCVI_LATENT_KEY] = scanvi_query.get_latent_representation(adata)

    adata = add_scanvi_predictions(
        adata, scanvi_query, insert_key=SCANVI_PREDICTIONS_KEY
    )

    if retrain or not qscanvi_path.exists():
        # save the reference model
        scanvi_query.save(qscanvi_path, overwrite=True)

    if plot_training:
        plot_scvi_training(scanvi_query.history)

    return scanvi_query, adata


def get_lbl8r_scvi(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "scvi_nobatch",
    plot_training: bool = False,
    **training_kwargs,
):
    """
    Get scVI model and latent representation for `LBL8R` model. Note that `batch_key=None`
    Just a wrapper for `get_trained_scvi` with `batch_key=None`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCVI_nobatch`.
    plot_training : bool
        Whether to plot training. Default is `False`.
    **training_kwargs : dict
        Additional arguments to pass to `scvi.model.SCVI.train`.

    Returns
    -------
    SCVI
        scVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    # just call get_trained_scvi with batch_key=None
    vae, adata = get_trained_scvi(
        adata,
        labels_key=labels_key,
        batch_key=None,
        model_path=model_path,
        retrain=retrain,
        model_name=model_name,
        plot_training=plot_training,
    )

    return vae, adata


def prep_lbl8r_adata(
    adata: AnnData,
    vae: SCVI | None = None,
    pca_key: str | None = None,
    labels_key: str = "cell_type",
) -> AnnData:
    """
    make an adata with embeddings in adata.X.  It if `vae is not None`
    it will use the `vae` model to make the latent representation.  Otherwise,
    if pca_key is not None, it will use the `adata.obsm{pca_key)' slot in the adata.
    If both are None, it will raise an error.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    vae : SCVI
        An scVI model. Default is `None`.
    pca_key : str
        Key for pca loadings. Default is `None`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    if vae is None and pca_key is None:
        raise ValueError("vae and pca_key cannot both be None")

    if vae is not None:
        # do i need an adata.copy() here?
        SCVI.setup_anndata(adata, labels_key=labels_key, batch_key=None)  # "dummy")

        latent_ad = make_latent_adata(vae, adata, return_dist=False)
        latent_ad.obsm[SCVI_LATENT_KEY] = latent_ad.X  # copy to obsm for convenience
        return latent_ad

    if pca_key is not None:
        loadings_ad = make_pc_loading_adata(adata, pca_key)
        return loadings_ad


def add_lbl8r_classifier(
    adata: AnnData,
    vae: SCVI,
    labels_key: str = "cell_type",
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "LBL8R_scVI_z",
    plot_training: bool = False,
    **training_kwargs,
) -> (SCANVI, AnnData):
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    vae : SCVI
        An scVI model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `LBL8R_scVI_z`.
    plot_training : bool
        Whether to plot training. Default is `False`.
    plot_preds : bool
        Whether to plot predictions. Default is `True`.
    **training_kwargs : dict
        Additional arguments to pass to `scvi.model.SCANVI.train`.

    Returns
    -------
    SCANVI
        scANVI model.
    AnnData
        Annotated data matrix with latent variables as X


    """
    scvi_epochs = 200
    batch_size = 512

    lbl8rscvi_path = model_path / model_name

    n_labels = len(adata.obs[labels_key].cat.categories)

    # 0. load/train model
    if lbl8rscvi_path.exists() and not retrain:
        vae_lbl8r_z = scviLBL8R.load(lbl8rscvi_path.as_posix(), adata.copy())

    else:
        vae_lbl8r_z = scviLBL8R.from_scvi_model(
            vae, adata.copy(), n_labels=n_labels, return_dist=False
        )
        vae_lbl8r_z.train(
            max_epochs=scvi_epochs,
            train_size=0.85,
            batch_size=batch_size,
            early_stopping=True,
        )
        latent_ad = vae_lbl8r_z.adata

    # 1. add the predictions to the adata
    predictions_z = vae_lbl8r_z.predict(probs=False, soft=True)
    latent_ad = add_predictions_to_adata(
        latent_ad, predictions_z, insert_key="pred", pred_key="label"
    )

    if retrain or not lbl8rscvi_path.exists():
        # save the reference model
        vae_lbl8r_z.save(lbl8rscvi_path, overwrite=True)

    if plot_training:
        plot_lbl8r_training(vae_lbl8r_z.history)

    return vae_lbl8r_z, latent_ad


def get_lbl8r(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "lbl8r",
    plot_training: bool = False,
    **training_kwargs,
):
    """ """
    PRED_KEY = "pred"

    lbl8r_path = model_path / model_name
    labels_key = labels_key
    n_labels = len(adata.obs[labels_key].cat.categories)

    lbl8r_epochs = 200
    batch_size = 512

    # not sure I need this step
    LBL8R.setup_anndata(adata, labels_key=labels_key)

    # 1. load/train model
    if lbl8r_path.exists() and not retrain:
        lat_lbl8r = LBL8R.load(lbl8r_path, adata.copy())
    else:
        lat_lbl8r = LBL8R(adata, n_labels=n_labels)
        lat_lbl8r.train(
            max_epochs=lbl8r_epochs,
            train_size=0.85,
            batch_size=batch_size,
            early_stopping=True,
            **training_kwargs,
        )

    # 1. add the predictions to the adata
    predictions_z = lat_lbl8r.predict(probs=False, soft=True)
    loadings_ad = add_predictions_to_adata(
        adata, predictions_z, insert_key="pred", pred_key="label"
    )

    if retrain or not lbl8r_path.exists():
        # save the reference model
        lat_lbl8r.save(lbl8r_path, overwrite=True)

    if plot_training:
        plot_lbl8r_training(lat_lbl8r.history)

    return lat_lbl8r, adata


# TODO:  add a flag to return predictions only rather than updating the adata?
def query_lbl8r(
    adata: AnnData,
    labelator: LBL8R,
    labels_key: str = "cell_type",
) -> AnnData:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labelator : scviLBL8R, pcaLBL8R, etc
        An classification model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    # labelator.setup_anndata(adata, labels_key=labels_key)  # "dummy")

    predictions = labelator.predict(adata, probs=False, soft=True)
    adata = add_predictions_to_adata(
        adata, predictions, insert_key="pred", pred_key="label"
    )

    return adata


# TODO: modularize things better so the pca/scvi versions call same code
def get_pca_lbl8r(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "LBL8R_pca",
    plot_training: bool = False,
    **training_kwargs,
):
    """ """
    pca_lbl8r, adata = get_lbl8r(
        adata,
        labels_key=labels_key,
        model_path=model_path,
        retrain=retrain,
        model_name=model_name,
        plot_training=plot_training,
        **training_kwargs,
    )

    return pca_lbl8r, adata

    # PRED_KEY = "pred"

    # lbl8rpca_path = model_path / model_name
    # labels_key = labels_key
    # n_labels = len(loadings_ad.obs[labels_key].cat.categories)

    # lbl8r_epochs = 200
    # batch_size = 512

    # # not sure I need this step
    # LBL8R.setup_anndata(loadings_ad, labels_key=labels_key)

    # # 1. load/train model
    # if lbl8rpca_path.exists() and not retrain:
    #     pca_lbl8r = LBL8R.load(lbl8rpca_path, loadings_ad.copy())
    # else:
    #     pca_lbl8r = LBL8R(loadings_ad, n_labels=n_labels)
    #     pca_lbl8r.train(
    #         max_epochs=lbl8r_epochs,
    #         train_size=0.85,
    #         batch_size=batch_size,
    #         early_stopping=True,
    #     )

    # # 1. add the predictions to the adata
    # predictions_z = pca_lbl8r.predict(probs=False, soft=True)
    # loadings_ad = add_predictions_to_adata(
    #     loadings_ad, predictions_z, insert_key="pred", pred_key="label"
    # )

    # if retrain or not lbl8rpca_path.exists():
    #     # save the reference model
    #     pca_lbl8r.save(lbl8rpca_path, overwrite=True)

    # if plot_training:
    #     plot_lbl8r_training(pca_lbl8r.history)

    # return pca_lbl8r, loadings_ad


# def prep_lbl8r_adata(
#     adata: AnnData,
#     vae: SCVI,
#     labels_key: str = "cell_type",
# ) -> AnnData:
#     """
#     Attach a classifier and prep adata for scVI LBL8R model

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data matrix.
#     vae : SCVI
#         An scVI model.
#     labels_key : str
#         Key for cell type labels. Default is `cell_type`.

#     Returns
#     -------
#     AnnData
#         Annotated data matrix with latent variables as X


#     """
#     # do i need an adata.copy() here?
#     SCVI.setup_anndata(adata, labels_key=labels_key, batch_key=None)  # "dummy")

#     latent_ad = make_latent_adata(vae, adata, return_dist=False)
#     latent_ad.obsm[SCVI_LATENT_KEY] = latent_ad.X  # copy to obsm for convenience

#     return latent_ad
