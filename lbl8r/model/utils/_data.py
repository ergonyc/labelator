import dataclasses
import pandas as pd
from pathlib import Path
import anndata as ad

# from scanpy.pp import pca
from ..._constants import OUT, H5


@dataclasses.dataclass
class Adata:
    """
    Adata class for storing and optionally backing AnnData objects.
    """

    path: str
    name: str
    is_backed: bool = False
    _adata: ad.AnnData = dataclasses.field(default=None, init=False, repr=False)

    def __init__(self, adata_path: Path, is_backed: bool = False):
        self.path = adata_path.stem
        self.name = adata_path.name
        self.is_backed = is_backed
        self._adata_path = adata_path

    @property
    def adata(self):
        if self._adata is None:
            if self.is_backed:
                self._adata = ad.read_h5ad(self._adata_path, backed="r+")
            else:
                self._adata = ad.read_h5ad(self._adata_path)
        return self._adata

    def export(self, out_path: Path):
        """
        Write adata to disk.
        """
        if out_path.suffix != ".h5ad":
            out_path = out_path.with_suffix(".h5ad")

        self.path = out_path.stem
        self.name = out_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.adata.write(out_path)

    def update(self, adata: ad.AnnData):
        """
        Update the adata object.
        """
        self._adata = adata


def export_ouput_adatas(adata, fname, out_data_path):
    """
    Export adata to disk.
    """
    adata.write(out_data_path / fname)


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


def transfer_pcs(train_ad: ad.AnnData, test_ad: ad.AnnData) -> ad.AnnData:
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


def sparsify_adata(adata: ad.AnnData):
    """replace the adata.X with a sparse matrix

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.

    Returns
    -------
    AnnData
        Annotated data matrix with sparse matrix.
    """
    import scipy.sparse as sparse

    adata.X = sparse.csr_matrix(adata.X)
    return adata


def export_ouput_adata(adata: ad.AnnData, file_name: str, out_path: Path):
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


def make_pc_loading_adata(adata: ad.AnnData, pca_key: str = "X_pca"):
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
        loading_adata = ad.AnnData(adata.obsm[pca_key])
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
