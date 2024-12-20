import pickle
from pathlib import Path
from anndata import AnnData
from numpy import ndarray
import pandas as pd
from ._pca import compute_pcs


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


def save_genes(genes: list[str], path: Path):
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


def archive_genes(ad: AnnData, model_path: Path):
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
    save_genes(genes, model_path)


# def save_predictions(preds: pd.DataFrame, model_path: Path):
#     """
#     Save predictions to feather file.

#     Parameters
#     ----------
#     preds : pandas DataFrame
#         Predictions.
#     model_path : Path
#         Path to the model.

#     """
#     preds_path = model_path / f"predictions.feather"
#     preds.to_feather(preds_path)
#     print(f"wrote: {preds_path}")


# def load_predictions(model_path: Path) -> pd.DataFrame:
#     """
#     Load predictions from feather file.

#     Parameters
#     ----------
#     model_path : Path
#         Path to the model.

#     Returns
#     -------
#     preds : pandas DataFrame
#         Predictions.

#     """
#     preds_path = model_path / f"predictions.feather"

#     if preds_path.exists():
#         preds = pd.read_feather(preds_path)
#         print(f"loaded n={len(preds)} predictions from {preds_path}")
#         return preds
#     else:
#         print(f"no predictions found at {preds_path}")
#         return None


def save_predictions(
    preds: pd.DataFrame, model_path: Path, compression: str = "snappy"
):
    """
    Save predictions to a Parquet file.

    Parameters
    ----------
    preds : pandas DataFrame
        Predictions.
    model_path : Path
        Path to the model.
    compression : str, optional
        Compression algorithm to use (e.g., 'snappy', 'gzip', 'zstd'). Default is 'snappy'.

    """
    preds_path = model_path / f"predictions.parquet"
    preds.to_parquet(preds_path, compression=compression)
    print(f"wrote: {preds_path} with compression={compression}")


def load_predictions(model_path: Path) -> pd.DataFrame:
    """
    Load predictions from a Parquet file.

    Parameters
    ----------
    model_path : Path
        Path to the model.

    Returns
    -------
    preds : pandas DataFrame
        Predictions, or None if the file does not exist.

    """
    preds_path = model_path / f"predictions.parquet"

    if preds_path.exists():
        preds = pd.read_parquet(preds_path)
        print(f"loaded n={len(preds)} predictions from {preds_path}")
        return preds
    else:
        print(f"no predictions found at {preds_path}")
        return None


def model_exists(model_path: Path) -> bool:
    """
    Check if model exists.

    Parameters
    ----------
    model_path : Path
        Path to the model.

    Returns
    -------
    bool
        True if model exists.

    """
    if model_path.is_file():
        return True

    if "xgb" in model_path.name:
        model_arch = model_path / "xgb.json"
    else:
        model_arch = model_path / "model.pt"

    # is_file is probably not necessary
    return model_arch.is_file()
