## functions for generating and archiving artifacts: figures, models, adatas

from dataclasses import dataclass
import anndata as ad
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from scvi.model import SCVI, SCANVI
from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from .._lbl8r import LBL8R
from ._data import Adata


@dataclass
class Model:
    """
    Model class for storing models.
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
        self, model, path, name, vae=None, labels_key="cell_type", label_encoder=None
    ):
        self.model = model
        self.path = path
        self.name = name
        self.vae = vae
        self.labels_key = labels_key
        self.label_encoder = label_encoder

    # def export(self, out_path: Path):
    #     """
    #     Write adata to disk.
    #     """
    #     if out_path.suffix != ".h5ad":
    #         out_path = out_path.with_suffix(".h5ad")

    #     self.path = out_path.stem
    #     self.name = out_path.name
    #     if not self.path.exists():
    #         self.path.mkdir()

    #     self.adata.write(out_path)

    # if isinstance(model, Booster):
    # bst_path = (
    #     model_path / model_name
    #     if model_name.endswith(".json")
    #     else model_path / f"{model_name}.json"
    # )
    #     # save the reference model
    #     bst.save_model(bst_path)


# @dataclasses.dataclass
# class Model:
#     """
#     Model class for storing models.
#     """

#     model: torch.nn.Module
#     model_path: str
#     model_name: str


# @dataclasses.dataclass
# class adata:
#     """
#     Adata class for storing adatas.
#     """

#     adata: ad.AnnData
#     adata_path: str
#     adata_name: str


# def make_expression_artifacat(adata,model, out_data_path):
#     train_ad = ad.read_h5ad(out_data_path / train_filen.name.replace(H5, OUT + H5))
#     exp_train_ad = make_scvi_normalized_adata(scvi_query, train_ad)

#     # In[ ]:
#     export_ouput_adata(exp_train_ad, train_filen.name.replace(CNT, EXPR), out_data_path)
#     del exp_train_ad, train_ad

#     test_ad = ad.read_h5ad(out_data_path / test_filen.name.replace(H5, OUT + H5))

#     # In[ ]:
#     # reset the cell_type_key before exporting
#     test_ad.obs[cell_type_key] = test_ad.obs["ground_truth"]
#     exp_test_ad = make_scvi_normalized_adata(scvi_query, test_ad)
#     export_ouput_adata(exp_test_ad, test_filen.name.replace(CNT, EXPR), out_data_path)
