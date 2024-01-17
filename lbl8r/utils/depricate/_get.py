from anndata import AnnData
from scvi.model import SCVI, SCANVI
from pathlib import Path


from ...model._lbl8r import (
    LBL8R,
    get_lbl8r,
    get_trained_scvi,
)

from ...model._scvi import (
    get_trained_scanvi,
    get_query_scvi,
    get_query_scanvi,
)

from ...model._xgb import (
    get_xgb,
)


from ...model.utils._data import Adata
from ...model.utils._Model import Model

from lbl8r._constants import *

PRED_KEY = "label"
INSERT_KEY = "pred"
# Model = SCVI | SCANVI | LBL8R | Booster


def get_model(
    adata: Adata,
    model_name: str,
    model_path: Path | str,
    model: SCVI | SCANVI | LBL8R | None = None,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    retrain: bool = False,
    **training_kwargs,
) -> (Model, Adata):
    """
    Load a model from the model_path

    Parameters
    ----------
    model_name : str
        Name of the model.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.

    Returns
    -------
    SCVI | SCANVI | LBL8R | Booster
        A model object.

    """

    model_path = model_path / model_name
    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    ad = adata.adata

    if model_name.startswith("scvi"):
        model = get_trained_scvi(
            adata.adata,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("scanvi"):
        if model is None:
            ValueError("SCANVI model must be passed")

        model = get_trained_scanvi(
            adata.adata,
            vae=model,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("lbl8r"):
        model = get_lbl8r(
            ad,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("xgb"):
        model = get_xgb(
            ad,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("query_scvi"):
        if model is None:
            ValueError("SCANVI model must be passed")

        model = get_query_scvi(
            ad,
            vae=model,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("query_scanvi"):
        if model is None:
            ValueError("SCANVI model must be passed")

        model = get_query_scanvi(
            ad,
            scanvi_model=model,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    else:
        raise ValueError(f"unknown model_name {model_name}")

    return model
