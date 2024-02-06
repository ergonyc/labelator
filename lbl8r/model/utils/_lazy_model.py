from dataclasses import dataclass, field
from scvi.model import SCVI, SCANVI
from numpy import ndarray
import pandas as pd

# from xgboost import Booster
# from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from .._lbl8r import LBL8R
from .._xgb import XGB
from ._artifact import load_pcs, load_genes
from dataclasses import dataclass, field
from pathlib import Path
from scvi.model import SCVI, SCANVI
from .._lbl8r import LBL8R
from .._xgb import XGB

# TODO: make this class handle the laoding / saving of models.


@dataclass
class LazyModel:
    """
    LazyModel class for storing models + metadata.
    """

    model_path: Path
    model: SCVI | SCANVI | LBL8R | XGB | None = None
    _name: str = field(init=False)
    _type: str = field(init=False)
    _helper: list[str] | None = field(default=None, init=False, repr=True)
    _model: SCVI | SCANVI | LBL8R | XGB | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        self._name = self.model_path.name
        self.path = self.model_path.parent
        self._infer_type_and_helper()

    def _infer_type_and_helper(self):
        # Infer type and helper based on the model name (_name)
        model_name = self._name
        if model_name.startswith("scanvi"):
            self._type = "scanvi"
            self._helper = ["scvi", "query_scvi", "query_scanvi"]
        elif model_name.endswith("_xgb"):
            self._type = "xgb"
            if "scvi" in model_name:
                self._helper = ["scvi"]
        else:  # Assuming default case is LBL8R
            self._type = "lbl8r"
            if "scvi" in model_name:
                self._helper = ["scvi"]

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

    @type.setter
    def type(self, model_type: str):
        if model_type not in ["scanvi", "lbl8r", "xgb"]:
            raise ValueError("model_type must be one of: 'scanvi', 'lbl8r', 'xgb'")
        self._type = model_type

    def load_model(self):
        # load model
        if self.type == "scanvi":
            self._model = SCANVI.load(self.model_path)
        elif self.type == "lbl8r":
            self._model = LBL8R.load(self.model_path)
        elif self.type == "xgb":
            self._model = XGB.load(self.model_path)
        else:
            raise ValueError(f"model_type must be one of: 'scanvi', 'lbl8r', 'xgb'")


@dataclass
class ModelSet:
    """
    Wrapper for model class for storing models + metadata.
    """

    model: dict[str, LazyModel]
    path: Path | str
    labels_key: str = "cell_type"
    _prepped: bool = field(default=False, init=False, repr=False)
    _pcs: ndarray | None = field(default=None, init=False, repr=False)
    _default: str | None = field(default=None, init=False, repr=False)
    _genes: list[str] | None = field(default_factory=list, init=False, repr=False)
    predictions: dict[str, pd.DataFrame] = field(
        default_factory=dict, init=True, repr=False
    )
    report: dict[str, dict] = field(default_factory=dict, init=True, repr=False)
    # _labels_key: str = field(init=False, repr=False)

    def __post_init__(self):
        # Ensure path is always a Path object
        self.path = Path(self.path)
        # load saved pcs if they exist
        self._pcs = load_pcs(self.path)
        # print(f"loaded pcs: {self.pcs}")
        print("post init load_genes")
        self._genes = load_genes(self.path)

    def add_model(self, mods: dict[str, LazyModel]):
        for name, model in mods.items():
            if name in self.model.keys():
                raise ValueError(f"Model name {name} already exists in model group")
            self.model[name] = model
        # self.model.update(mods)

    # @property
    # def labels_key(self):
    #     return self._labels_key

    # @labels_key.setter
    # def labels_key(self, key: str):
    #     self._labels_key = key

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

    @property
    def pcs(self):
        if self._pcs is None:
            self._pcs = load_pcs(self.path)
        return self._pcs

    @pcs.setter
    def pcs(self, pcs: ndarray):
        self._pcs = pcs

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, model_name: str):
        if model_name not in self.model.keys():
            raise ValueError(f"model_name must be one of: {self.model.keys()}")
        self._default = model_name
