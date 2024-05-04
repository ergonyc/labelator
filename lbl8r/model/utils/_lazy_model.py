from dataclasses import dataclass, field
from scvi.model import SCVI, SCANVI
from numpy import ndarray
import pandas as pd

# from xgboost import Booster
# from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from anndata import AnnData

from .._lbl8r import LBL8R
from .._xgb import XGB
from ._artifact import load_genes
from dataclasses import dataclass, field
from pathlib import Path
from scvi.model import SCVI, SCANVI
from .._lbl8r import LBL8R
from .._xgb import XGB

from ..._constants import SCVI_LATENT_KEY, SCANVI_LATENT_KEY, PCA_KEY

# TODO: make this class handle the laoding / saving of models.


@dataclass
class LazyModel:
    """
    LazyModel class for storing models + metadata.
    """

    model_path: Path
    _model: SCVI | SCANVI | LBL8R | XGB | None = None
    _name: str = field(init=False)
    _type: str = field(init=False)
    # _helper: list[str] | None = field(default=None, init=False, repr=True)
    _loaded: bool = field(default=False, init=False, repr=False)
    # _model: SCVI | SCANVI | LBL8R | XGB | None = field(
    #     default=None, init=False, repr=False
    # )

    def __post_init__(self):
        self._name = self.model_path.name
        self.path = self.model_path
        self._infer_type()
        if self.model is not None:
            self._model = self.model
            self._loaded = True

    def _infer_type(self):
        # Infer type and helper based on the model name (_name)
        model_name = self._name

        if model_name.startswith("scanvi"):
            self._type = "scanvi"
        elif model_name.endswith("_xgb"):
            self._type = "xgb"
        elif model_name == "scvi":
            self._type = "scvi"
        else:  # Assuming default case is LBL8R
            self._type = "lbl8r"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def loaded(self):
        return self._loaded

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, model_name: str):
        self._name = model_name
        self._infer_type_and_helper()

    @property
    def type(self):
        return self._type

    # @type.setter
    # def type(self, model_type: str):
    #     if model_type not in ["scanvi","scvi", "lbl8r", "xgb"]:
    #         raise ValueError("model_type must be one of: 'scanvi','scvi', 'lbl8r', 'xgb'")
    #     self._type = model_type

    def load_model(self, adata: AnnData):
        # load model
        if self.type == "scanvi":
            self._model = SCANVI.load(self.model_path, adata.copy())
        elif self.type == "scvi":
            self._model = SCVI.load(self.model_path, adata.copy())
        elif self.type == "lbl8r":
            self._model = LBL8R.load(self.model_path, adata.copy())
        elif self.type == "xgb":
            model = XGB()
            model.load(self.model_path)
            self._model = model
        else:
            raise ValueError(
                f"model_type must be one of: 'scanvi','scvi', 'lbl8r', 'xgb'"
            )
        self._loaded = True


@dataclass
class ModelSet:
    """
    Wrapper for model class for storing models + metadata.
    """

    model: dict[str, LazyModel]
    path: Path | str
    labels_key: str = "cell_type"
    batch_key: str | None = None
    _prepped: bool = field(default=False, init=False, repr=False)
    # _X_pca: ndarray | None = field(default=None, init=False, repr=False)
    _default: str | None = field(default=None, init=False, repr=False)
    _genes: list[str] | None = field(default_factory=list, init=False, repr=False)
    _basis: str | None = field(default=None, init=False, repr=False)
    _name: str | None = field(default=None, init=False, repr=True)

    predictions: pd.DataFrame | None = field(default=None, init=False, repr=False)

    # TODO: depricate report (only xbgoost)
    report: dict = field(default_factory=dict, init=False, repr=False)

    # _labels_key: str = field(init=False, repr=False)

    def __post_init__(self):
        # Ensure path is always a Path object
        self.path = Path(self.path)
        # load saved pcs if they exist
        self._name = self.path.name
        # if "pcs" in self.path.name or "raw" in self.path.name:
        #     print(f"pre init load_pcs: {self.path.name}")
        #     self._pcs = load_pcs(self.path)
        # print(f"loaded pcs: {self.pcs}")
        print("post init load_genes")
        self._genes = load_genes(self.path)

    def add_model(self, mods: dict[str, LazyModel]):
        for name, model in mods.items():
            if name in self.model.keys():
                raise ValueError(f"Model name {name} already exists in model group")
            self.model[name] = model
        # self.model.update(mods)

    @property
    def name(self):
        if self._name is None:
            self._name = self.path.name
        return self._name

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, key: str):
        if key not in [PCA_KEY, SCVI_LATENT_KEY, SCANVI_LATENT_KEY]:
            print(
                f"basis must be one of: {PCA_KEY},{SCVI_LATENT_KEY}, {SCANVI_LATENT_KEY}"
            )
            print(f"setting basis to default: {PCA_KEY}")
            self._basis = PCA_KEY
        else:
            self._basis = key

    @property
    def genes(self):
        return self._genes

    # might not need the setter
    @genes.setter
    def genes(self, genes: list[str]):
        print(f"setting genes: n={len(genes)}")
        self._genes = genes

    @property
    def prepped(self):
        return self._prepped

    @prepped.setter
    def prepped(self, value: bool):
        self._prepped = value

    # @property
    # def X_pca(self):
    #     if self._X_pca is None:
    #         # self._X_pca = load_pcs(self.path)
    #         print(f"no X_pca.  needs to be set")
    #     return self._X_pca

    # @X_pca.setter
    # def X_pca(self, x_pca: ndarray):
    #     self._X_pca = x_pca
    #     # dump_pcs(x_pca, self.path)

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, model_name: str):
        if model_name not in self.model.keys():
            raise ValueError(f"model_name must be one of: {self.model.keys()}")
        self._default = model_name

    # def load_model(self, model_name: str, adata: AnnData):
    #     self.model[model_name].load_model(adata)

    # elif basis == SCVI_LATENT_KEY:
    # ad.obsm[SCVI_MDE_KEY] = mde(ad.obsm[SCVI_LATENT_KEY], device=device)

    #     elif basis == SCANVI_LATENT_KEY:
    # ad.obsm[SCANVI_MDE_KEY] = mde(ad.obsm[SCANVI_LATENT_KEY], device=device)
