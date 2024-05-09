import scanpy as sc
from numpy import ndarray
from ._timing import Timing
import numpy as np
from pathlib import Path
from ..._constants import N_PCS


@Timing(prefix="compute_pcs: ")
def compute_pcs(adata: sc.AnnData, n_pcs: int = N_PCS) -> ndarray:
    """
    Compute principal components.  This function is a wrapper around scanpy.pp.pca.  Only used if we don't already have them saved. (e.g. scvi_expr training data)

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_pcs : int
        Number of principal components to compute.

    Returns
    -------
    pcs : ndarray
        Principal components.

    """
    bdata = adata.copy()
    print("compute_pcs - normalize_total")
    sc.pp.normalize_total(bdata, target_sum=1e4)
    print("compute_pcs - log1p")
    sc.pp.log1p(bdata)
    # print("compute_pcs - scale")
    # sc.pp.scale(bdata, max_value=10)
    print("compute_pcs - pca")
    sc.pp.pca(bdata, n_comps=n_pcs)

    pcs = bdata.varm["PCs"].copy()
    X_pca = bdata.obsm["X_pca"].copy()
    # update adata in place
    # adata.uns["PCs"] = pcs
    # adata.obsm["X_pca"] = bdata.obsm["X_pca"]
    # adata.varm["PCs"] = pcs

    return pcs, X_pca


@Timing(prefix="transfer_pca: ")
def transfer_pca(
    adata: sc.AnnData,
    pcs: ndarray,
) -> sc.AnnData:
    """
    Transfer principal components.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    pcs : ndarray
        Principal components.

    Returns
    -------
    bdata : AnnData
        Annotated data matrix with pca transferred.

    """

    bdata = adata.copy()
    # scale values.
    sc.pp.normalize_total(bdata, target_sum=1e4)
    sc.pp.log1p(bdata)
    # sc.pp.scale(bdata, max_value=10)

    # zero center
    # Calculate the mean of each column
    col_means = bdata.X.sum(axis=0)
    col_means /= bdata.X.shape[0]

    if bdata.X.shape[0] < 600_000:
        X_pca = (bdata.X - col_means) @ pcs
    else:
        chunk_size = 10_000
        n_chunks = bdata.X.shape[0] // chunk_size

        X_pca = np.zeros((bdata.X.shape[0], pcs.shape[1]))
        for chunk in range(n_chunks):
            start = chunk * chunk_size
            end = start + chunk_size
            X_pca[start:end] = (bdata.X[start:end] - col_means) @ pcs

        # now do the last chunk
        start = n_chunks * chunk_size
        X_pca[start:] = (bdata.X[start:] - col_means) @ pcs

    return X_pca


# @Timing(prefix="transfer_pca: ")
# def transfer_pca(
#     adata: sc.AnnData,
#     pcs: np.ndarray,
# ) -> sc.AnnData:
#     """
#     Transfer principal components.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data matrix.
#     pcs : ndarray
#         Principal components (eigenvectors) of training set.

#     Returns
#     -------
#     x_pca : ndarray
#         projection of data onto PCs.

#     """
#     bdata = adata.copy()
#     # scale values.
#     print("compute_pcs - normalize_total")
#     sc.pp.normalize_total(bdata, target_sum=1e4)
#     print("compute_pcs - log1p")
#     sc.pp.log1p(bdata)
#     X = bdata.X
#     del bdata
#     # Calculate the mean of each column
#     col_means = X.sum(axis=0)
#     col_means /= X.shape[0]

#     # now chunk the X to get the x_pca.
#     chunk_size = 10_000
#     n_chunks = X.shape[0] // chunk_size

#     X_pca = np.zeros((X.shape[0], pcs.shape[1]))
#     for chunk in range(n_chunks):
#         start = chunk * chunk_size
#         end = start + chunk_size
#         X_pca[start:end] = (X[start:end] - col_means) @ pcs

#     # now do the last chunk
#     start = n_chunks * chunk_size
#     X_pca[start:] = (X[start:] - col_means) @ pcs

#     # # Subtract the column means from each column
#     # # col_means is 1x3 matrix; we use np.array to ensure proper broadcasting
#     # adjusted_X = X - csr_matrix(np.ones((X.shape[0], 1))) @ csr_matrix(col_means)

#     #     del bdata
#     #     x_pca = np.matmul(X, pcs)
#     #     # x_pca = bdata.X @ pcs
#     return X_pca


def load_pcs(pcs_path: Path) -> np.ndarray:
    """
    Load principal components from adata.

    Parameters
    ----------
    pcs_path : Path
        Path to save the PCs.

    Returns
    -------
    pcs : np.ndarray
        Principal components.

    """

    pcs_path = pcs_path / f"PCs.npy"
    if pcs_path.exists():
        return np.load(pcs_path)
    else:
        print(f"no PCs found at {pcs_path}")
        return None


def load_x_pca(xpca_path: Path, xpca_name: str = "X_pca.npy") -> np.ndarray:
    """
    Load principal components from adata.

    Parameters
    ----------
    xpca_path : Path
        Path to save the X_pca.

    Returns
    -------
    X_pca : np.ndarray
        Principal components.

    """

    xpca_path = xpca_path / xpca_name
    if xpca_path.exists():
        return np.load(xpca_path)
    else:
        print(f"no X_pca found at {xpca_path}")
        return None


def dump_pcs(pcs: np.ndarray, pcs_path: Path):
    """
    Save principal components (eigenvectors) of data.

    Parameters
    ----------
    pcs : ndarray
        Principal components.
    pcs_path : Path
        Path to save the PCs.


    """
    if not pcs_path.exists():
        pcs_path.mkdir(parents=True, exist_ok=True)

    pcs_path = pcs_path / f"PCs.npy"
    np.save(pcs_path, pcs)


def dump_x_pca(x_pca: np.ndarray, xpca_path: Path, xpca_name: str = "X_pca.npy"):
    """
    Save principal components representation of data.

    Parameters
    ----------
    x_pca : ndarray
        Principal components.
    pcs_path : Path
        Path to save the PCs.
    """
    if not xpca_path.exists():
        xpca_path.mkdir(parents=True, exist_ok=True)

    xpca_path = xpca_path / xpca_name
    np.save(xpca_path, x_pca)
