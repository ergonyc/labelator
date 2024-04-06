import scanpy as sc
from numpy import ndarray
from ._timing import Timing


@Timing(prefix="compute_pcs: ")
def compute_pcs(adata: sc.AnnData, n_pcs: int = 50) -> ndarray:
    """
    Compute principal components.

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
    print("compute_pcs - scale")
    sc.pp.scale(bdata, max_value=10)
    print("compute_pcs - pca")
    sc.pp.pca(bdata, n_comps=n_pcs)

    pcs = bdata.varm["PCs"].copy()
    # update adata in place
    adata.uns["PCs"] = pcs
    adata.obsm["X_pca"] = bdata.obsm["X_pca"]
    adata.varm["PCs"] = pcs

    return pcs


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
    sc.pp.scale(bdata, max_value=10)

    loadings = bdata.X @ pcs
    bdata = adata.copy()
    bdata.obsm["X_pca"] = loadings
    # update adata in place
    bdata.uns["PCs"] = pcs
    bdata.varm["PCs"] = pcs

    return bdata
