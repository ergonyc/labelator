from anndata import AnnData
from scvi.model import SCANVI
import pandas as pd

from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from .constants import OUT, H5
from .modules._xgb import test_xgboost
from .utils import merge_into_obs
from .models._lbl8r import LBL8R

PRED_KEY = "label"
INSERT_KEY = "pred"


def query_model(
    adata: AnnData,
    model: LBL8R | SCANVI | Booster,
    label_encoder: LabelEncoder | None = None,
    labels_key: str = "cell_type",
) -> AnnData:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model : LBL8R | SCANVI | Booster
        An classification model.
    label_encoder : LabelEncoder
        The label encoder.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    if isinstance(model, LBL8R):
        return query_lbl8r(adata, model, labels_key=labels_key)
    elif isinstance(model, SCANVI):
        return query_scanvi(adata, model, insert_key=labels_key)
    elif isinstance(model, Booster):
        return query_xgb(adata, model, label_encoder)
    else:
        raise ValueError(f"model {model} is not a valid model")


def query_scanvi(ad: AnnData, model: SCANVI, insert_key: str = "label"):
    """
    Get the "soft" and label predictions from a SCANVI model,
    and then add into the ad.obs

    Parameters
    ----------
    ad : ad.AnnData
        AnnData object to add the predictions to
    model : SCANVI
        SCANVI model to use to get the predictions
    Returns
    -------
    ad.AnnData
        AnnData object with the predictions added

    """

    predictions = model.predict(ad, soft=True)
    predictions[insert_key] = model.predict(ad, soft=False)

    obs = ad.obs

    # TODO: call merge_into_obs
    if set(predictions.columns) & set(obs.columns):
        ValueError("Predictions and obs have overlapping columns")
        return ad

    obs = pd.merge(obs, predictions, left_index=True, right_index=True, how="left")

    ad.obs = obs
    return ad


def query_xgb(
    adata: AnnData,
    bst: Booster,
    label_encoder: LabelEncoder,
) -> (AnnData, dict):
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labelator : Booster
        An XGBoost classification model.
    label_encoder : LabelEncoder
        The label encoder.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    predictions, report = test_xgboost(bst, adata, label_encoder)
    # loadings_ad = add_predictions_to_adata(
    #     adata, predictions, insert_key=INSERT_KEY, pred_key=PRED_KEY
    # )
    adata = merge_into_obs(adata, predictions)

    return adata, report


# TODO:  add a flag to return predictions only rather than updating the adata?
def query_lbl8r(
    adata: AnnData,
    labelator: LBL8R,
    labels_key: str = "cell_type",
) -> AnnData:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labelator : scviLBL8R, pcaLBL8R, etc
        An classification model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    # labelator.setup_anndata(adata, labels_key=labels_key)  # "dummy")

    predictions = labelator.predict(adata, probs=False, soft=True)
    # loadings_ad = add_predictions_to_adata(
    #     adata, predictions, insert_key=INSERT_KEY, pred_key=PRED_KEY
    # )
    adata = merge_into_obs(adata, predictions)
    return adata
