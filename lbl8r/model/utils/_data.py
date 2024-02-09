import dataclasses
import pandas as pd
from pathlib import Path
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse

# from scanpy.pp import pca
from ..._constants import OUT, H5, EMB, EXPR, CNT, PCS, VAE, CELL_TYPE_KEY


@dataclasses.dataclass
class Adata:
    """
    Adata class for storing and optionally backing AnnData objects.
    """

    path: str
    name: str
    is_backed: bool = False
    _adata: ad.AnnData = dataclasses.field(default=None, init=False, repr=False)
    _out_suffix: str = dataclasses.field(default=None, init=False, repr=False)
    _subdir: str = dataclasses.field(default=None, init=False, repr=False)
    _loaded: bool = dataclasses.field(default=True, init=True, repr=True)
    _adata_path: Path = dataclasses.field(default=None, init=False, repr=False)
    _labels_key: str = dataclasses.field(default=None, init=False, repr=False)
    _ground_truth_key: str = dataclasses.field(default=None, init=False, repr=False)
    _archive_path: Path = dataclasses.field(default=None, init=False, repr=False)

    def __init__(self, adata_path: Path, is_backed: bool = False):
        if adata_path is None:
            # instantiate an empty adata object
            self.path = None
            self.name = "empty"
            self.is_backed = False
        else:
            self.path = adata_path.parent
            self.name = adata_path.name
            self.is_backed = is_backed

    @property
    def adata(self):
        if self.path is None:
            return None

        if self._adata is None:
            if self.is_backed:
                print(f"WARNING::: untested loading backed adata: {self.adata_path}")
                self._adata = ad.read_h5ad(self.adata_path, backed="r+")
            else:
                self._adata = ad.read_h5ad(self.adata_path)
        self._loaded = True

        self.ground_truth_key = CELL_TYPE_KEY
        return self._adata

    @property
    def adata_path(self):
        _adata_path = Path(self.path) / self.name
        return _adata_path

    @property
    def loaded(self):
        return self._adata is not None

    @property
    def labels_key(self):
        return self._labels_key

    @labels_key.setter
    def labels_key(self, labels_key: str):
        self._labels_key = labels_key

    @property
    def ground_truth_key(self):
        return self._ground_truth_key

    @ground_truth_key.setter
    def ground_truth_key(self, ground_truth_key: str):
        if self._adata is None:
            print("adata not loaded")
            return None

        if ground_truth_key in self._adata.obs_keys():
            self._ground_truth_key = ground_truth_key

    @property
    def archive_path(self):
        if self._archive_path is None and self._adata_path is None:
            print("No adata.  Just a placeholder")
            return None
        else:
            return self._archive_path

    @archive_path.setter
    def archive_path(self, archive_path: str | Path):
        self._archive_path = Path(archive_path)

    def export(self, out_path: Path | None = None):
        """
        Write adata to disk.
        """
        if out_path is not None:
            self.archive_path = out_path

        out_path = (
            self.archive_path
            if self._subdir is None
            else self.archive_path / self._subdir
        )

        if self._out_suffix is not None:
            out_path = out_path / self.name.replace(H5, self._out_suffix + OUT + H5)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.adata.write(out_path)
            print(f"wrote: {out_path}")
        else:
            print("This model doesn't need to export AnnData")

    def update(self, adata: ad.AnnData):
        """
        Update the adata object.
        """
        self._adata = adata

    def set_output(self, model_name: str):
        """
        Set the output name suffix and subdir based on model_name
        """

        # set path_subdir based on model_name
        if model_name.startswith("scvi"):
            self._subdir = "LBL8R" + VAE
        elif model_name == "scanvi":
            self._subdir = "SCANVI"
        elif model_name.startswith("scanvi_"):
            self._subdir = "SCANV_batch_eq"
        elif model_name.startswith("xgb"):
            self._subdir = "XGB"
        elif model_name.startswith("lbl8r"):
            # got to LBL8R_pcs, or LBL8R (e2e)
            if model_name.endswith("_pcs"):
                self._subdir = "LBL8R" + PCS
            else:
                self._subdir = "LBL8R"
        else:
            self._subdir = "ERROR"

        # set suffix based on model_name
        if model_name.endswith(EMB):
            self._out_suffix = EMB
        elif model_name.endswith(EXPR):
            self._out_suffix = EXPR
        elif model_name.endswith(EXPR + PCS):
            self._out_suffix = EXPR
        elif model_name.endswith(CNT + PCS):
            self._out_suffix = PCS
        elif model_name.startswith("scanvi"):
            self._out_suffix = "_scanvi"
        elif model_name.endswith("raw_cnt"):
            self._out_suffix = ""
        else:
            self._out_suffix = ""  # eventually we will disable the other outputs


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


# TODO: remove ref_ad input
def transfer_pcs(
    query_ad: ad.AnnData,
    ref_ad: ad.AnnData | None = None,
    pcs: np.ndarray | None = None,
) -> ad.AnnData:
    """Transfer PCs from training data to get "loadings" (`X_pca`)

    Parameters
    ----------
    query_ad
        AnnData object with "test" data in X
    ref_ad
        AnnData object for training with PCs in varm["PCs"]
    pcs
        Principal components ndarray from ref_ad.

    Returns
    -------
    AnnData
        AnnData object with PCs in varm["PCs"] and "loadings" in obsm["X_pca"]
    """

    # copy the old variables if they exist (i.e. raw count pcs)
    if "X_pca" in query_ad.obsm_keys():
        X_pca = query_ad.obsm.pop("X_pca")
        query_ad.obsm["_X_pca"] = X_pca

    if "PCs" in query_ad.varm_keys():
        PCs = query_ad.varm.pop("PCs")
        print("saving raw PCs to query_ad.uns['_PCs']")
        query_ad.uns["_PCs"] = PCs

    if "pca" in query_ad.uns_keys():
        pca_dict = query_ad.uns.pop("pca")
        query_ad.uns["_pca"] = pca_dict
        _ = query_ad.uns.pop("_scvi_uuid", None)
        _ = query_ad.uns.pop("_scvi_manager_uuid", None)

    # transfer PCs from ref_ad to query_ad
    if pcs is not None:
        query_ad.uns["PCs"] = pcs
    elif ref_ad is not None:
        if "PCs" in ref_ad.varm_keys():
            print("transfering PCs from ref_ad to query_ad")
            query_ad.uns["PCs"] = ref_ad.varm["PCs"].copy()
        elif "PCs" in ref_ad.uns_keys():
            print("transfering PCs from ref_ad to query_ad")
            query_ad.uns["PCs"] = ref_ad.uns["PCs"].copy()
        else:
            raise ValueError("No PCs found in ref_ad")
    else:
        raise ValueError("No PCs available")

    # compute loadings
    # TODO: check shape of X and PCs for compatibility
    query_ad.obsm["X_pca"] = query_ad.X @ query_ad.uns["PCs"]
    # update the uns dictionary. there migth be a reason/way to compute this relative to ad_out
    # query_ad.uns.update(ref_ad.uns.copy())

    return query_ad


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
    if pca_key in adata.obsm_keys():
        # already have the loadings...
        loading_adata = ad.AnnData(adata.obsm[pca_key])
    else:  #
        ValueError("Need to get PCA loadings first")
        print(f"doing nothing: {pca_key} not in {adata.obsm_keys()}")

        return adata

    var_names = [f"pc_{i}" for i in range(loading_adata.shape[1])]

    loading_adata.obs_names = adata.obs_names.copy()
    loading_adata.obs = adata.obs.copy()
    loading_adata.var_names = var_names
    loading_adata.obsm = adata.obsm.copy()
    loading_adata.uns = adata.uns.copy()
    _ = loading_adata.uns.pop("_scvi_uuid", None)
    _ = loading_adata.uns.pop("_scvi_manager_uuid", None)
    # hold the PCS from varm in uns for transferring to query
    if "PCs" in adata.varm_keys():
        loading_adata.uns["PCs"] = adata.varm["PCs"].copy()
    elif "PCs" in adata.uns_keys():
        loading_adata.uns["PCs"] = adata.uns["PCs"].copy()
    else:
        raise ValueError("No PCs found in adata")

    return loading_adata


def prep_target_genes(ad: ad.AnnData, target_genes: list[str]) -> ad.AnnData:
    """
    Expand AnnData object to include all target_genes.  Missing target_genes will be added as zeros.
    """
    # Identify missing variables
    missing_vars = list(set(target_genes) - set(ad.var_names))
    # Create a dataframe/matrix of zeros for missing variables
    if len(missing_vars) > 0:
        if issparse(ad.X):
            zeros = csr_matrix((ad.n_obs, len(missing_vars)))
        elif isinstance(ad.X, np.ndarray):
            zeros = np.zeros((ad.n_obs, len(missing_vars)))
        else:
            raise ValueError("X must be a numpy array or a sparse matrix")

        # Create an AnnData object for the missing variables
        missing_ad = ad.AnnData(
            X=zeros, var=pd.DataFrame(index=missing_vars), obs=ad.obs
        )
        # Concatenate the original and the missing AnnData objects along the variables axis
        expanded_ad = ad.concat([ad, missing_ad], axis=1, join="outer")
    else:
        expanded_ad = ad.copy()
    # Ensure the order of variables matches all_vars
    return expanded_ad[:, target_genes]
