import pickle
from pathlib import Path
from anndata import AnnData
from numpy import ndarray
import scanpy as sc


def dump_pcs(pcs: ndarray, model_path: Path):
    """
    Save principal components to adata.

    Parameters
    ----------
    pcs : ndarray
        Principal components.
    model_path : Path
        Path to the model.


    """
    pcs_path = model_path / f"pcs.pkl"
    with open(pcs_path, "wb") as f:
        pickle.dump(pcs, f)
    print(f"wrote: {pcs_path}")


def load_pcs(model_path: Path) -> ndarray:
    """
    Load principal components from adata.

    Parameters
    ----------
    model_path : Path
        Path to the model.

    Returns
    -------
    pcs : np.ndarray
        Principal components.

    """

    pcs_path = model_path / f"pcs.pkl"

    if pcs_path.exists():
        with open(pcs_path, "rb") as f:
            pcs = pickle.load(f)
        return pcs
    else:
        print(f"no pcs found at {pcs_path}")
        return None


def extract_pcs(ad: AnnData) -> ndarray:
    """
    Extract principal components from adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    Returns
    -------
    pcs : np.ndarray
        Principal components.
    """
    if "PCs" in ad.varm_keys():
        print("getting PCs from varm")
        pcs = ad.varm["PCs"].copy()
    elif "PCs" in ad.uns_keys():
        print("transfering PCs from ref_ad to query_ad")
        pcs = ad.uns["PCs"].copy()
    else:
        sc.pp.pca(ad)
        pcs = ad.varm["PCs"].copy()

    return pcs


def save_pcs(ad: AnnData, model_path: Path):
    """
    Archive principal components to adata.

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix.
    model_path : Path
        Path to the model.

    """
    pcs = extract_pcs(ad)
    print(pcs.shape)
    dump_pcs(pcs, model_path)
