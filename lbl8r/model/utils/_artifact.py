import pickle
from pathlib import Path
from anndata import AnnData
from numpy import ndarray
import scanpy as sc
import pandas as pd


def _dump_pkl(obj, path: Path):
    """
    Dump object to pickle file.

    Parameters
    ----------
    obj : Any
        Object to dump.
    path : Path
        Path to save object.

    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"wrote: {path}")


def _load_pkl(path: Path):
    """
    Load object from pickle file.

    Parameters
    ----------
    path : Path
        Path to load object.

    Returns
    -------
    obj : Any
        Loaded object.

    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


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
    _dump_pkl(pcs, pcs_path)
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
        return _load_pkl(pcs_path)
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
    dump_pcs(pcs, model_path)


def dump_genes(genes: list[str], path: Path):
    """
    Save genes to pickle file.

    Parameters
    ----------
    genes : list[str]
        List of gene names.
    path : Path
        Path to save genes.

    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    genes_path = path / f"genes.pkl"
    _dump_pkl(genes, genes_path)
    # print(f"wrote: N={len(genes)} to {genes_path}")


def load_genes(path: Path) -> list[str]:
    """
    Load genes from pickle file.

    Parameters
    ----------
    path : Path
        Path to load genes.

    Returns
    -------
    genes : list[str]
        List of gene names.

    """
    genes_path = path / f"genes.pkl"

    if genes_path.exists():
        genes = _load_pkl(genes_path)
        # print(f"loaded n={len(genes)} genes from {genes_path}")
        return genes
    else:
        print(f"no genes found at {genes_path}")
        return None


def save_genes(ad: AnnData, model_path: Path):
    """
    Archive genes to adata.

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix.
    model_path : Path
        Path to the model.

    """
    genes = ad.var_names.tolist()
    dump_genes(genes, model_path)


def save_predictions(preds: pd.DataFrame, model_path: Path):
    """
    Save predictions to feather file.

    Parameters
    ----------
    preds : pandas DataFrame
        Predictions.
    model_path : Path
        Path to the model.

    """
    preds_path = model_path / f"predictions.feather"
    preds.to_feather(preds_path)
    print(f"wrote: {preds_path}")


def load_predictions(model_path: Path) -> pd.DataFrame:
    """
    Load predictions from feather file.

    Parameters
    ----------
    model_path : Path
        Path to the model.

    Returns
    -------
    preds : pandas DataFrame
        Predictions.

    """
    preds_path = model_path / f"predictions.feather"

    if preds_path.exists():
        preds = pd.read_feather(preds_path)
        print(f"loaded n={len(preds)} predictions from {preds_path}")
        return preds
    else:
        print(f"no predictions found at {preds_path}")
        return None
