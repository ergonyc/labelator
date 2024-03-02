from scvi.model import SCVI, SCANVI
from anndata import AnnData

from pathlib import Path
import pandas as pd
import numpy as np

from .utils._timing import Timing
from .utils._artifact import model_exists

from .._constants import *

# from .._constants import SCVI_LATENT_KEY_Z, SCVI_LATENT_KEY_MU_VAR


def prep_latent_z_adata(
    adata: AnnData,
    vae: SCVI,
    labels_key: str = "cell_type",
) -> AnnData:
    """
    make an adata with VAE latent z embedding in adata.X.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    vae : SCVI
        An scVI model.
    labels_key : str
        Key for pca loadings. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """
    adata = adata.copy()
    SCVI.setup_anndata(adata, labels_key=labels_key, batch_key=None)  # "dummy")

    latent_ad = make_latent_adata(adata, scvi_model=vae, return_dist=False)
    latent_ad.obsm[SCVI_LATENT_KEY] = (
        latent_ad.X
    )  # copy to obsm for convenience (doubles size 🫤 but useful for plot_embedding)
    return latent_ad


# TODO: add save and load flags so we can use the functions and NOT overwrite on accident
@Timing(prefix="model_name")
def get_trained_scvi(
    adata: AnnData,
    labels_key: str = "cell_type",
    # batch_key: str | None = None,
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "scvi",
    **training_kwargs,
) -> tuple[SCVI, AnnData]:
    """
    Get scVI model and add latent representation to adata

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
        Name of the model. Default is `SCVI`.
    training_kwargs : dict
        Keyword arguments for training. implicitly contains batch_key : str
        Key for batch labels. Default is `None`.


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

    # this should work to extract batch_key for batch corrected scvi
    batch_key = training_kwargs.pop("batch_key", None)
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
    if model_exists(scvi_path) and not retrain:
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

    if retrain or not model_exists(scvi_path):
        # save the reference model
        vae.save(scvi_path, overwrite=True)

    return vae, adata


@Timing(prefix="model_name")
def get_trained_scanvi(
    adata: AnnData,
    vae: SCVI | None = None,
    labels_key: str = "cell_type",
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "scanvi",
    **training_kwargs,
) -> tuple[SCANVI, AnnData]:
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

    Returns
    -------
    SCANVI
        scANVI model.
    AnnData
        Annotated dat


    a matrix with latent variables.

    """
    scvi_epochs = 200
    batch_size = 512

    # n_labels = len(adata.obs[labels_key].cat.categories)
    # this should work to extract batch_key for batch corrected scvi
    batch_key = training_kwargs.pop("batch_key", None)

    if vae is None:
        # get the scVI model.  Note the desired batch_key needs to be passed
        # BUG: this will overwrite the SCVI model... need to fix
        scvi_model_name = "SCVI" + model_name
        vae, adata = get_trained_scvi(
            adata,
            labels_key=labels_key,
            batch_key=batch_key,
            model_path=model_path,
            retrain=False,
            model_name=scvi_model_name,
            **training_kwargs,
        )

    scanvi_path = model_path / model_name
    if model_exists(scanvi_path) and not retrain:
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

    if retrain or not model_exists(scanvi_path):
        # save the reference model
        scanvi_model.save(scanvi_path, overwrite=True)

    return scanvi_model, adata


@Timing(prefix="model_name")
def get_query_scvi(
    adata: AnnData,
    vae: SCVI | str,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    model_path: Path = Path.cwd(),
    batch_eq: bool = False,
    retrain: bool = False,
    model_name: str = "query_scvi",
    **training_kwargs,
) -> tuple[SCVI, AnnData]:
    """
    Get scVI model via `scarches` surgery and add latent representation to adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    vae : SCVI
        An scVI model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    batch_key : str
        Key for batch labels. Default is `None`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    batch_eq : bool
        Whether to use batch equalization. Default is `False`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCVI_query`.
    training_kwargs : dict
        Keyword arguments for training.


    Returns
    -------
    SCVI
        scVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    surgery_epochs = 150

    qscvi_path = model_path / model_name
    # failing here... ?!?
    SCVI.prepare_query_anndata(adata, vae)
    # the query model might exist if we are not batch correcting... need to fix...
    if batch_eq:
        # just use the vae.
        if isinstance(vae, SCVI):
            scvi_query = vae
        else:
            scvi_query = SCVI.load(vae, adata)

    elif model_exists(qscvi_path) and not retrain:
        scvi_query = SCVI.load(qscvi_path.as_posix(), adata)
    else:
        scvi_query = SCVI.load_query_data(adata, vae)
        scvi_query.train(
            max_epochs=surgery_epochs,
            train_size=0.85,
            early_stopping=True,
            plan_kwargs=dict(weight_decay=0.0),
            # datasplitter_kwargs=dict(num_workers=15),
            # **training_kwargs,
        )

    adata.obsm[SCVI_LATENT_KEY] = scvi_query.get_latent_representation(adata)
    if retrain or not model_exists(qscvi_path):
        scvi_query.save(qscvi_path, overwrite=True)

    return scvi_query, adata


@Timing(prefix="model_name")
def get_query_scanvi(
    adata: AnnData,
    scanvi_model: SCANVI,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    model_path: Path = Path.cwd(),
    batch_eq: bool = False,
    retrain: bool = False,
    model_name: str = "query_scanvi",
    **training_kwargs,
) -> tuple[SCANVI, AnnData]:
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
    batch_eq : bool
        Whether to use batch equalization. Default is `False`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCANVI_query`.

    Returns
    -------
    SCANVI
        scANVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    surgery_epochs = 150

    qscanvi_path = model_path / model_name

    # should the reference be teh scvi_model or the scanvi_model?
    SCANVI.prepare_query_anndata(adata, scanvi_model)

    # the query model might exist if we are not batch correcting... need to fix...
    if batch_eq:
        # just use the scanvi_model.
        if isinstance(scanvi_model, SCANVI):
            scvi_query = scanvi_model
        else:
            scvi_query = SCVI.load(scanvi_model, adata)

    elif model_exists(qscanvi_path) and not retrain:
        scanvi_query = SCANVI.load(qscanvi_path.as_posix(), adata)
    else:
        scanvi_query = SCANVI.load_query_data(adata, scanvi_model)
        scanvi_query.train(
            max_epochs=surgery_epochs,
            train_size=0.85,
            early_stopping=True,
            plan_kwargs=dict(weight_decay=0.0),
            # datasplitter_kwargs=dict(num_workers=15),
        )

    adata.obsm[SCVI_LATENT_KEY] = scanvi_query.get_latent_representation(adata)

    if retrain or not model_exists(qscanvi_path):
        # save the reference model
        scanvi_query.save(qscanvi_path, overwrite=True)

    return scanvi_query, adata


def query_scanvi(ad: AnnData, model: SCANVI) -> pd.DataFrame:
    """
    Get the "soft" and label predictions from a SCANVI model,
    and then add into the ad.obs

    Parameters
    ----------
    ad : ad.AnnData
        AnnData object to add the predictions to
    model : SCANVI
        SCANVI model to use to get the predictions
    Returns
    -------
    ad.AnnData
        AnnData object with the predictions added

    """
    insert_key = "label"
    predictions = model.predict(ad, soft=True)
    predictions[insert_key] = model.predict(ad, soft=False)

    return predictions
    # ad = merge_into_obs(ad, predictions)
    # return ad


def add_latent_obsm(ad: AnnData, model: SCVI | SCANVI) -> AnnData:
    """
    Add the latent representation from a scVI model into the ad.obsm

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix.
    model : SCVI | SCANVI
        An scVI or scANVI model.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables.

    """
    key = SCVI_LATENT_KEY if isinstance(model, SCVI) else SCANVI_LATENT_KEY

    ad.obsm[key] = model.get_latent_representation(ad)
    return ad


def make_latent_adata(
    adata: AnnData, scvi_model: SCVI | None = None, return_dist: bool = True
):
    """
    make an `AnnData` object with the latent representation from an scvi model.  Use the SCVI_LATENT_KEY
    to access the latent representation by default.  If `scvi_model` is not provided, then use the model to
    generate the latent representation.  If `return_dist` is True, then return the distribution: i.e. both the
    mean and the variance latents.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    scvi_model : SCVI
        An scvi model.
    return_dist : bool
        Whether to return the mean or the distribution. Default is `True`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables.

    """

    latent_key = SCVI_LATENT_KEY_Z if return_dist else SCVI_LATENT_KEY

    # try and copy latent from obsm
    if latent_key in adata.obsm_keys():
        # create latent adata and var_names
        latent_adata = AnnData(adata.obsm[latent_key])
        if return_dist:
            var_names = [f"zm_{i}" for i in range(latent_adata.shape[1] // 2)] + [
                f"zv_{i}" for i in range(latent_adata.shape[1] // 2)
            ]
        else:
            var_names = [f"z_{i}" for i in range(latent_adata.shape[1])]

    else:  # get latent representations from the model
        if scvi_model is not None:
            if return_dist:
                qzm, qzv = scvi_model.get_latent_representation(
                    adata, give_mean=False, return_dist=return_dist
                )
                latent_adata = AnnData(np.concatenate([qzm, qzv], axis=1))
                var_names = [f"zm_{i}" for i in range(qzm.shape[1])] + [
                    f"zv_{i}" for i in range(qzv.shape[1])
                ]
            else:
                latent_adata = AnnData(
                    scvi_model.get_latent_representation(adata, give_mean=True)
                )
                var_names = [f"z_{i}" for i in range(latent_adata.shape[1])]

        else:
            ValueError(
                f"need to provide a `scvi_model` or have adata.obsm[{latent_key}]"
                " to get latent representation"
            )

    latent_adata.obs_names = adata.obs_names.copy()
    latent_adata.obs = adata.obs.copy()
    latent_adata.var_names = var_names
    latent_adata.obsm = adata.obsm.copy()

    latent_adata.uns = {}

    return latent_adata


def make_scvi_normalized_adata(
    scvi_model: SCVI,
    adata: AnnData,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
):
    """
    Gets normalized expression from scvi model

    Parameters
    ----------
    scvi_model : SCVI
        An scvi model.
    adata : AnnData
        Annotated data matrix.
    labels_key : str
        Labels key. Default is `cell_type`.
    batch_key : str
        Batch key. Default is `None`.

    Returns
    -------
    AnnData
        Annotated data matrix with normalized expression.

    """

    exp_adata = adata.copy()

    if "X_pca" in exp_adata.obsm_keys():
        X_pca = exp_adata.obsm.pop("X_pca")
        exp_adata.obsm["_X_pca"] = X_pca

    if "PCs" in exp_adata.varm_keys():
        PCs = exp_adata.varm.pop("PCs")
        print("adding raw PCs to exp_adata")
        exp_adata.varm["_PCs"] = PCs

    if "pca" in exp_adata.uns_keys():
        pca_dict = exp_adata.uns.pop("pca")
        exp_adata.uns["_pca"] = pca_dict
        _ = exp_adata.uns.pop("_scvi_uuid", None)
        _ = exp_adata.uns.pop("_scvi_manager_uuid", None)

    scvi_model.setup_anndata(exp_adata, labels_key=labels_key, batch_key=batch_key)
    denoised = scvi_model.get_normalized_expression(
        exp_adata,
        library_size=1e4,
        return_numpy=True,
    )

    exp_adata.X = denoised

    exp_adata = add_latent_obsm(exp_adata, scvi_model)

    return exp_adata
