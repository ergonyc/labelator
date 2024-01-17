from anndata import AnnData
from scvi.model import SCANVI

from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from anndata import AnnData

from lbl8r.model import LBL8R, query_lbl8r, query_scanvi, query_xgb


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
