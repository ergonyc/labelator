import dataclasses
from typing import Any

from scvi.model import SCVI, SCANVI
from numpy import ndarray

# from xgboost import Booster
# from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from .._lbl8r import LBL8R
from .._xgb import XGB
from ._pcs import load_pcs

# TODO: make this class handle the laoding / saving of models.


@dataclasses.dataclass
class LazyModel:
    """
    LazyModel class for storing models + metadata.
    """

    # TODO: make a __repr__ method which will print the properties rather than the _attributes

    _model_path: str
    _name: str
    _type: str
    _name: str
    _helper: Any = None
    _model: SCVI | SCANVI | LBL8R | XGB | None = None
    # _adata_path: str

    def __init__(
        self,
        model_path: Path,
        model: SCVI | SCANVI | LBL8R | XGB | None = None,
    ):
        self.path = model_path.parent
        self.name = model_path.name
        self._model_path = model_path
        self._model = model

        def __post_init__(self):
            self.name = self.name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, model_name: str):
        if model_name is not None:
            self._name = model_name

            # infer type
            if model_name.startswith("scanvi"):
                self.type = "scanvi"
                self._helper = ["scvi", "query_scvi", "query_scanvi"]
            elif model_name.endswith("_xgb"):
                self.type = "xgb"
                if "scvi" in model_name:
                    self._helper = ["scvi"]

            else:  # could be REPR or CNT model
                self.type = "lbl8r"
                if "scvi" in model_name:
                    self._helper = ["scvi"]

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, model_type: str):
        if model_type not in ["scanvi", "lbl8r", "xgb"]:
            raise ValueError(f"model_type must be one of: 'scanvi', 'lbl8r', 'xgb'")
        self._type = model_type

    @property
    def model(self):
        if self.path is None:
            return None

        if self._model is None:
            self.load_model()

        return self._model

    @property
    def model_path(self):
        return self._model_path

    def load_model(self):
        # load model
        if self.type == "scanvi":
            self._model = SCANVI.load(self.model_path)
        elif self.type == "lbl8r":
            self._model = LBL8R.load(self.model_path)
        elif self.type == "xgb":
            print(f"loading xgb model from: {self.model_path}")
            # model = XGB(path=self.model_path)
            self._model = XGB.load(self.model_path)
        else:
            raise ValueError(f"model_type must be one of: 'scanvi', 'lbl8r', 'xgb'")


@dataclasses.dataclass
class ModelSet:
    """
    Wrapper for model class for storing models + metadata.
    """

    model: dict[str, LazyModel]
    path: str
    labels_key: str | None = "cell_type"
    _prepped: bool = False
    _pcs: ndarray | None = None

    def __init__(
        self,
        mods: dict[str, LazyModel],
        path: str | Path,
        labels_key="cell_type",
    ):
        self.model = mods
        self.path = Path(path) if isinstance(path, str) else path
        self.labels_key = labels_key

    # # TODO: __repr__ method
    # def __repr__(self):
    #     repr_str = super().__repr__().replace("ModelSet", "")
    #     repr_str += f"ModelSet({self.prepped}, {self.pcs}"
    #     return repr_str

    def __post_init__(self):
        # load saved pcs if they exist
        self._pcs = load_pcs(self.path)
        # print(f"loaded pcs: {self.pcs}")

    def add_model(self, mods: dict[str, LazyModel]):
        for name, model in mods.items():
            if name in self.mods.keys():
                raise ValueError(f"Model name {name} already exists in model group")
            self.model[name] = model
        # self.model.update(mods)

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
