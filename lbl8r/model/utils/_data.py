import dataclasses
import pandas as pd
from pathlib import Path
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from ._pca import transfer_pca, compute_pcs, dump_pcs
from ._mde import mde
from ._device import get_usable_device

# from scanpy.pp import pca
from ..._constants import (
    OUT,
    H5,
    EMB,
    EXPR,
    CNT,
    PCS,
    VAE,
    CELL_TYPE_KEY,
    BATCH_EQUALIZED,
    RAW_PC_MODEL_NAME,
    RAW_COUNT_MODEL_NAME,
    SCVI_LATENT_KEY,
    SCANVI_LATENT_KEY,
    SCVI_MDE_KEY,
    SCANVI_MDE_KEY,
    MDE_KEY,
    PCA_KEY,
)


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
    _loaded: bool = dataclasses.field(default=False, init=True, repr=True)
    _adata_path: Path = dataclasses.field(default=None, init=False, repr=False)
    _labels_key: str = dataclasses.field(default=None, init=False, repr=False)
    _ground_truth_key: str = dataclasses.field(default=None, init=False, repr=False)
    _archive_path: Path = dataclasses.field(default=None, init=False, repr=False)
    _artifact_path: Path = dataclasses.field(default=None, init=False, repr=False)
    _predictions: Path = dataclasses.field(default=None, init=False, repr=False)
    _pcs: Path = dataclasses.field(default=None, init=False, repr=False)
    _X_pca: Path = dataclasses.field(default=None, init=False, repr=False)
    _X_mde: Path = dataclasses.field(default=None, init=False, repr=False)
    _X_scvi: Path = dataclasses.field(default=None, init=False, repr=False)
    _X_scanvi: Path = dataclasses.field(default=None, init=False, repr=False)
    _X_scvi_mde: Path = dataclasses.field(default=None, init=False, repr=False)
    _X_scanvi_mde: Path = dataclasses.field(default=None, init=False, repr=False)

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
        self._adata_path = Path(self.path) / self.name
        return self._adata_path

    @property
    def pcs(self):
        if self._pcs is None:
            # try load from disk
            pcs_path = self.artifact_path / "PCs.npy"
            if pcs_path.exists():
                self._pcs = np.load(pcs_path)
            # elif "PCs" in self.adata.varm_keys():
            #     self._pcs = self.adata.varm["PCs"]
            #     dump_pcs(self._pcs, self.artifact_path)
            else:
                print(f"no PCs found at {pcs_path}")
                # TODO: add compute PCs
                self._pcs = None

        return self._pcs

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, predictions: pd.DataFrame):
        self._predictions = predictions

    @property
    def X_pca(self):
        if self._X_pca is None:
            # try load from disk
            x_pca_path = (
                self.artifact_path / f"X_pca_{self.name.replace('h5ad', 'npy')}"
            )
            if x_pca_path.exists():
                self._X_pca = np.load(x_pca_path)
            else:
                # compute!!
                # see if we have PCs and if so transfer them
                if self.pcs is None:
                    print(f"computing pca for {self.name}")
                    self._pcs, self._X_pca = compute_pcs(self.adata)
                    # save the PCs and X_pca
                    dump_pcs(self._pcs, self.artifact_path)
                    dump_x_repr(self._X_pca, self.artifact_path, x_name=x_pca_path.name)
                else:
                    print(f"transferring PCs to {self.name}")
                    self._X_pca = transfer_pca(self.adata, self.pcs)
                    # save the X_pca
                    dump_x_repr(self._X_pca, self.artifact_path, x_name=x_pca_path.name)

        return self._X_pca

    @property
    def X_mde(self):
        if self._X_mde is None:
            # try load from disk
            x_mde_path = (
                self.artifact_path / f"X_mde_{self.name.replace('h5ad', 'npy')}"
            )
            if x_mde_path.exists():
                self._X_mde = np.load(x_mde_path)
            else:
                # compute!!
                print(f"computing mde for {self.name} pca latent")
                self._X_mde = mde(self.X_pca)
                dump_x_repr(self._X_mde, self.artifact_path, x_name=x_mde_path.name)

        return self._X_mde

    @property
    def X_scvi(self):
        if self._X_scvi is None:
            # try load from disk
            x_path = self.artifact_path / f"X_scvi_{self.name.replace('h5ad', 'npy')}"
            if x_path.exists():
                self._X_scvi = np.load(x_path)
            else:
                # get from adata
                if SCVI_LATENT_KEY in self.adata.obsm_keys():
                    self._X_scvi = self.adata.obsm[SCVI_LATENT_KEY]
                    dump_x_repr(self._X_scvi, self.artifact_path, x_name=x_path.name)
                else:
                    print(f"no scvi latent found in adata.obs[SCVI_LATENT_KEY]")

        return self._X_scvi

    @property
    def X_scvi_mde(self):
        if self._X_scvi_mde is None:
            # try load from disk
            x_mde_path = (
                self.artifact_path / f"X_scvi_mde_{self.name.replace('h5ad', 'npy')}"
            )
            if x_mde_path.exists():
                self._X_scvi_mde = np.load(x_mde_path)
            else:
                # compute!!
                print(f"computing mde for {self.name} scvi latent")
                self._X_scvi_mde = mde(self.X_scvi)
                dump_x_repr(
                    self._X_scvi_mde, self.artifact_path, x_name=x_mde_path.name
                )

        return self._X_scvi_mde

    @property
    def X_scanvi(self):
        if self._X_scanvi is None:
            # try load from disk
            x_path = self.artifact_path / f"X_scanvi_{self.name.replace('h5ad', 'npy')}"
            if x_path.exists():
                self._X_scanvi = np.load(x_path)
            else:
                # get from adata
                if SCANVI_LATENT_KEY in self.adata.obsm_keys():
                    self._X_scanvi = self.adata.obsm[SCANVI_LATENT_KEY]
                    dump_x_repr(self._X_scanvi, self.artifact_path, x_name=x_path.name)
                else:
                    print(f"no scanvi latent found in adata.obs[SCANVI_LATENT_KEY]")

        return self._X_scanvi

    @property
    def X_scanvi_mde(self):
        if self._X_scanvi_mde is None:
            # try load from disk
            x_mde_path = (
                self.artifact_path / f"X_scanvi_mde_{self.name.replace('h5ad', 'npy')}"
            )
            if x_mde_path.exists():
                self._X_scanvi_mde = np.load(x_mde_path)
            else:
                # compute!!
                print(f"computing mde for {self.name} scanvi latent")
                self._X_scanvi_mde = mde(self.X_scanvi)
                dump_x_repr(
                    self._X_scanvi_mde, self.artifact_path, x_name=x_mde_path.name
                )

        return self._X_scanvi_mde

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
        return self._archive_path

    @archive_path.setter
    def archive_path(self, archive_path: str | Path):
        self._archive_path = Path(archive_path)
        if not self._archive_path.exists():
            self._archive_path.mkdir(parents=True, exist_ok=True)

    @property
    def artifact_path(self):
        return self.path if self._archive_path.stem == "count" else self._archive_path

    def export(self, out_path: Path | None = None):
        """
        Write adata to disk.
        """
        if out_path is not None:
            self.archive_path = out_path
        else:
            out_path = self.archive_path

        if self._out_suffix is not None:
            out_path = out_path / self.name.replace(H5, self._out_suffix + OUT + H5)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.adata.write(out_path)
            print(f"wrote: {out_path}")
        else:
            print("This model doesn't need to export AnnData")

        if self._predictions is not None:
            preds_path = (
                self.archive_path / f"predictions_{self.name.replace('h5ad','feather')}"
            )
            preds = self._predictions.reset_index()
            preds.to_feather(preds_path)
            print(f"wrote: {preds_path}")
        else:
            print("No predictions to export ? ")

    def update(self, adata: ad.AnnData):
        """
        Update the adata object.
        """
        self._adata = adata

    def set_output(self, model_name: str, batch_eq: bool = False):
        """
        Set the output name suffix and subdir based on model_name
        """
        # TODO: clean this up

        # if model_name in ["scvi_emb", "scvi_expr", "scvi_expr_pcs"]:
        #     self._subdir = "batch_eq" if batch_eq else "naive"
        # elif model_name in ["raw", "pcs"]:
        #     self._subdir = "count"
        # else:
        #     self._subdir = "ERROR"

        # set suffix based on model_name
        if model_name.endswith(EMB):
            self._out_suffix = EMB
        elif model_name.endswith(EXPR):
            self._out_suffix = EXPR
        elif model_name.endswith(EXPR + PCS):
            self._out_suffix = EXPR + PCS
        elif model_name == RAW_COUNT_MODEL_NAME:
            self._out_suffix = CNT
        elif model_name == RAW_PC_MODEL_NAME:
            self._out_suffix = CNT + PCS
        elif model_name.endswith("raw"):
            self._out_suffix = CNT
        else:
            self._out_suffix = ""  # eventually we will disable the other outputs


def dump_x_repr(x_repr: np.ndarray, x_path: Path, x_name: str):
    """
    dump artifact (X_repr) representation of data.

    Parameters
    ----------
    x_repr : ndarray
        representation of data.
    x_path : Path
        Path to save X_repr.
    x_name : str
        Name of the file.

    """
    if not x_path.exists():
        x_path.mkdir(parents=True, exist_ok=True)

    x_path = x_path / x_name
    np.save(x_path, x_repr)


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
    if pcs is None and ref_ad is not None:
        if "PCs" in ref_ad.varm_keys():
            print("transfering PCs from ref_ad to query_ad")
            pcs = ref_ad.varm["PCs"].copy()
        elif "PCs" in ref_ad.uns_keys():
            print("transfering PCs from ref_ad to query_ad")
            pcs = ref_ad.uns["PCs"].copy()
        else:
            raise ValueError("No PCs found in ref_ad")

    query_ad = transfer_pca(query_ad, pcs)

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


def make_pc_loading_adata(adata: ad.AnnData, x_pca: np.ndarray, pca_key: str = "X_pca"):
    """
    Makes adata with PCA loadings

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x_pca : np.ndarray
        data projected on Principal components.
    pca_key : str
        Key in `adata.obsm` where PCA loadings are stored. Default is `X_pca`.
        If `X_pca` is not in `adata.obsm`, then it will raise an error.

    Returns
    -------
    AnnData
        Annotated data matrix with PCA loadings.

    """

    loading_adata = ad.AnnData(x_pca)
    var_names = [f"pc_{i}" for i in range(loading_adata.shape[1])]

    loading_adata.obs_names = adata.obs_names.copy()
    loading_adata.obs = adata.obs.copy()
    loading_adata.var_names = var_names

    obsm = adata.obsm.copy()
    obsm[pca_key] = x_pca
    loading_adata.obsm = obsm

    loading_adata.uns = adata.uns.copy()
    _ = loading_adata.uns.pop("_scvi_uuid", None)
    _ = loading_adata.uns.pop("_scvi_manager_uuid", None)

    print("made `loading_adata` with PCA loadings")
    return loading_adata


def add_pc_loadings(adata: ad.AnnData, x_pca: np.ndarray, pca_key: str = "X_pca"):
    """
    Makes adata with PCA loadings

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    x_pca : np.ndarray
        data projected on Principal components.
    pca_key : str
        Key in `adata.obsm` where PCA loadings are stored. Default is `X_pca`.
        If `X_pca` is not in `adata.obsm`, then it will raise an error.

    Returns
    -------
    AnnData
        Annotated data matrix with PCA loadings.

    """

    adata.obsm[pca_key] = x_pca

    return adata


def add_mde_obsm(ad: ad.AnnData, X_mde: np.ndarray, basis: str = "X_pca") -> ad.AnnData:
    """
    Add the latent representation from a scVI model into the ad.obsm

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix.
    basis : str
        key to apply mde to. Could be: `X_pca`, `X_scVI`,`X_scANVI`

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables.

    """
    device = get_usable_device()

    if basis == PCA_KEY:
        ad.obsm[MDE_KEY] = mde(ad.obsm[PCA_KEY], device=device)

    elif basis == SCVI_LATENT_KEY:
        ad.obsm[SCVI_MDE_KEY] = mde(ad.obsm[SCVI_LATENT_KEY], device=device)

    elif basis == SCANVI_LATENT_KEY:
        ad.obsm[SCANVI_MDE_KEY] = mde(ad.obsm[SCANVI_LATENT_KEY], device=device)

    return ad


def prep_target_genes(adata: ad.AnnData, target_genes: list[str]) -> ad.AnnData:
    """
    Expand AnnData object to include all target_genes.  Missing target_genes will be added as zeros.
    """
    print("                prepping target genes")
    # Identify missing variables
    missing_vars = list(set(target_genes) - set(adata.var_names))
    # Create a dataframe/matrix of zeros for missing variables
    if len(missing_vars) > 0:
        if issparse(adata.X):
            zeros = csr_matrix((adata.n_obs, len(missing_vars)))
        elif isinstance(ad.X, np.ndarray):
            zeros = np.zeros((adata.n_obs, len(missing_vars)))
        else:
            raise ValueError("X must be a numpy array or a sparse matrix")

        # Create an AnnData object for the missing variables
        missing_ad = ad.AnnData(
            X=zeros, var=pd.DataFrame(index=missing_vars), obs=adata.obs
        )
        print(missing_ad)
        # Concatenate the original and the missing AnnData objects along the variables axis
        expanded_ad = ad.concat([adata, missing_ad], axis=1, join="outer")
        expanded_ad.obs = adata.obs.copy()
        print(expanded_ad)
    else:
        expanded_ad = adata.copy()
    # Ensure the order of variables matches all_vars
    return expanded_ad[:, target_genes]
