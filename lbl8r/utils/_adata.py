from anndata import AnnData
from scvi.model import SCVI
import pandas as pd
from pathlib import Path
import numpy as np

# from scanpy.pp import pca
from ..constants import OUT, H5


def make_latent_adata(scvi_model: SCVI, adata: AnnData, return_dist: bool = True):
    """

    Parameters
    ----------
    scvi_model : SCVI
        An scvi model.
    adata : AnnData
        Annotated data matrix.
    return_dist : bool
        Whether to return the mean or the distribution. Default is `True`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables.

    """

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

    latent_adata.obs_names = adata.obs_names.copy()
    latent_adata.obs = adata.obs.copy()
    latent_adata.var_names = var_names
    latent_adata.obsm = adata.obsm.copy()

    latent_adata.uns = {}
    print(latent_adata.shape)

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

    scvi_model.setup_anndata(adata, labels_key=labels_key, batch_key=batch_key)
    denoised = scvi_model.get_normalized_expression(
        adata,
        library_size=1e4,
        return_numpy=True,
    )

    exp_adata = AnnData(denoised)

    exp_adata.obs_names = adata.obs_names.copy()
    exp_adata.obs = adata.obs.copy()
    exp_adata.var = adata.var.copy()
    exp_adata.obsm = adata.obsm.copy()
    exp_adata.uns = adata.uns.copy()
    exp_adata.varm = adata.varm.copy()

    # rename keys for PCA from the raw data
    # better pattern:
    # val = exp_adata.obsm.pop(key, None)
    # if val is not None:
    #     exp_adata.obsm[f"_{key}"] = val

    if "X_pca" in exp_adata.obsm.keys():
        X_pca = exp_adata.obsm.pop("X_pca")
        exp_adata.obsm["_X_pca"] = X_pca

    if "PCs" in exp_adata.varm.keys():
        PCs = exp_adata.varm.pop("PCs")
        print("adding raw PCs to exp_adata")
        exp_adata.varm["_PCs"] = PCs

    if "pca" in exp_adata.uns.keys():
        pca_dict = exp_adata.uns.pop("pca")
        exp_adata.uns["_pca"] = pca_dict
        _ = exp_adata.uns.pop("_scvi_uuid", None)
        _ = exp_adata.uns.pop("_scvi_manager_uuid", None)

    return exp_adata


def make_pc_loading_adata(adata: AnnData, pca_key: str = "X_pca"):
    """
    Makes adata with PCA loadings

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    pca_key : str
        Key in `adata.obsm` where PCA loadings are stored. Default is `X_pca`.
        If `X_pca` is not in `adata.obsm`, then it will raise an error.

    Returns
    -------
    AnnData
        Annotated data matrix with PCA loadings.

    """
    if pca_key in adata.obsm.keys():
        # already have the loadings...
        loading_adata = AnnData(adata.obsm[pca_key])
    else:  #
        ValueError("Need to get PCA loadings first")
        print("doing nothing")
        return adata

    var_names = [f"pc_{i}" for i in range(loading_adata.shape[1])]

    loading_adata.obs_names = adata.obs_names.copy()
    loading_adata.obs = adata.obs.copy()
    loading_adata.var_names = var_names
    loading_adata.obsm = adata.obsm.copy()
    loading_adata.uns = adata.uns.copy()
    _ = loading_adata.uns.pop("_scvi_uuid", None)
    _ = loading_adata.uns.pop("_scvi_manager_uuid", None)

    print(loading_adata.shape)
    return loading_adata


def add_predictions_to_adata(adata, predictions, insert_key="pred", pred_key="label"):
    """
    Add the predictions to the adata object. Performs a merge in case shuffled
    data is in the predictions table.


    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    predictions : pd.DataFrame
        Pandas DataFrame of predictions.
    insert_key : str
        Key in `adata.obs` where predictions are stored. Default is `pred`.
    pred_key : str
        Key in `adata.obs` where cell types are stored. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with predictions.

    """

    obs = adata.obs
    if insert_key in obs.columns:
        # replace if its already there
        obs.drop(columns=[insert_key], inplace=True)

    adata.obs = pd.merge(
        obs, predictions[pred_key], left_index=True, right_index=True, how="left"
    ).rename(columns={pred_key: insert_key})

    return adata


def transfer_pcs(train_ad: AnnData, test_ad: AnnData) -> AnnData:
    """Transfer PCs from training data to get "loadings" (`X_pca`)

    Parameters
    ----------
    train_ad
        AnnData object for training with PCs in varm["PCs"]
    test_ad
        AnnData object with "test" data in X

    Returns
    -------
    AnnData
        AnnData object with PCs in varm["PCs"] and "loadings" in obsm["X_pca"]
    """

    if "X_pca" in test_ad.obsm.keys():
        X_pca = test_ad.obsm.pop("X_pca")
        test_ad.obsm["_X_pca"] = X_pca

    if "PCs" in test_ad.varm.keys():
        PCs = test_ad.varm.pop("PCs")
        print("saving raw PCs to test_ad")
        test_ad.varm["_PCs"] = PCs

    if "pca" in test_ad.uns.keys():
        pca_dict = test_ad.uns.pop("pca")
        test_ad.uns["_pca"] = pca_dict
        _ = test_ad.uns.pop("_scvi_uuid", None)
        _ = test_ad.uns.pop("_scvi_manager_uuid", None)

    test_ad.varm["PCs"] = train_ad.varm["PCs"].copy()
    # compute loadings
    test_ad.obsm["X_pca"] = test_ad.X @ test_ad.varm["PCs"]
    # update the uns dictionary. there migth be a reason/way to compute this relative to ad_out
    # test_ad.uns.update(train_ad.uns.copy())

    return test_ad


def merge_into_obs(adata, source_table, insert_keys=None, prefix=None):
    """
    Add the predictions to the adata object. Performs a merge in case shuffled


    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    source_table : pd.DataFrame
        Pandas DataFrame which has column to insert.
    insert_key : str
        Key in `adata.obs` where predictions are stored.
    prefix : str

    Returns
    -------
    AnnData
        Annotated data matrix with predictions.

    """

    insert_key = "pred"
    pred_key = "label"

    obs = adata.obs
    if insert_keys is None:
        df = source_table
        insert_keys = source_table.columns
    else:
        df = source_table[insert_keys]

    if any([k in obs.columns for k in insert_keys]):
        if prefix is None:
            prefix = "_"
        df = df.add_prefix(prefix)
        pred_key = "_label"

    adata.obs = pd.merge(obs, df, left_index=True, right_index=True, how="left").rename(
        columns={pred_key: insert_key}
    )

    return adata


def sparsify_adata(adata: AnnData):
    """replace the adata.X with a sparse matrix

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    Returns
    -------
    AnnData
        Annotated data matrix with sparse matrix.
    """
    import scipy.sparse as sparse

    adata.X = sparse.csr_matrix(adata.X)
    return adata


def export_ouput_adata(adata: AnnData, file_name: str, out_path: Path):
    """
    Export the AnnData object with the model name and file name appended to the file name.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model_name : str
        Name of the model.
    file_name : str
        Name of the file.
    data_path : Path
        Path to the data.


    """
    if not out_path.exists():
        out_path.mkdir()

    # if filename ends with _out.h5ad, strip the _out since we are adding back in
    file_name = file_name.replace(OUT, "").replace(H5, OUT + H5)

    adata.write_h5ad(out_path / file_name)

    print(f"wrote: {out_path / file_name}")
    return None
