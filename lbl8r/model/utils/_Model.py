from dataclasses import dataclass

from scvi.model import SCVI, SCANVI
from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from .._lbl8r import LBL8R


@dataclass
class Model:
    """
    Model class for storing models + metadata.
    """

    model: SCVI | SCANVI | LBL8R | Booster
    path: str
    name: str
    vae: SCVI | None = None
    labels_key: str | None = "cell_type"
    label_encoder: LabelEncoder | None = None
    q_vae: SCVI | None = None
    scanvi: SCANVI | None = None

    def __init__(
        self,
        model,
        path,
        name,
        vae=None,
        labels_key="cell_type",
        label_encoder=None,
    ):
        self.model = model
        self.path = path
        self.name = name
        self.vae = vae
        # TODO: shoudl the keys be in the Adata or Model dataclass?
        self.labels_key = labels_key
        self.label_encoder = label_encoder
