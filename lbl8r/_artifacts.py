## functions for generating and archiving artifacts: figures, models, adatas

import dataclasses
import anndata as ad
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from scvi.model import SCVI, SCANVI
from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from .models._lbl8r import LBL8R


@dataclasses.dataclass
class Figure:
    """
    Figure class for storing figures.
    """

    fig: plt.Figure
    fig_path: str

    def __init__(self, fig, fig_path):
        self.fig = fig
        self.fig_path = fig_path

    def savefig(self, adata, fig_kwargs):
        """
        Save figure to disk.
        """

        self.fig.savefig(self.fig_path, bbox_inches="tight")


@dataclasses.dataclass
class Model:
    """
    Model class for storing models.
    """

    model: SCVI | SCANVI | LBL8R
    path: str
    name: str

    def export(self, out_path: Path):
        """
        Write adata to disk.
        """
        if out_path.suffix != ".h5ad":
            out_path = out_path.with_suffix(".h5ad")

        self.path = out_path.stem
        self.name = out_path.name
        if not self.path.exists():
            self.path.mkdir()

        self.adata.write(out_path)


@dataclasses.dataclass
class Adata:
    """
    Adata class for storing adatas.
    """

    adata: ad.AnnData
    path: str
    name: str

    def __init__(self, adata_path: Path):
        self.path = adata_path.stem
        self.adata = ad.read_h5ad(adata_path)
        self.name = adata_path.name

    def export(self, out_path: Path):
        """
        Write adata to disk.
        """
        if out_path.suffix != ".h5ad":
            out_path = out_path.with_suffix(".h5ad")

        self.path = out_path.stem
        self.name = out_path.name
        if not self.path.exists():
            self.path.mkdir()

        self.adata.write(out_path)


@dataclasses.dataclass
class Artifact:
    """
    Artifacts class for storing artifacts.
    """

    model: Model
    adata: Adata
    fig: Figure


@dataclasses.dataclass
class Artifacts:
    """
    Artifacts class for storing artifacts.
    """

    artifacts: Artifact

    def __init__(self, artifacts):
        self.artifacts = artifacts

    def export_artifacts(self):
        """
        Export artifacts to disk.
        """
        for artifact in self.artifacts:
            if isinstance(artifact, Model):
                torch.save(artifact.model.state_dict(), artifact.model_path)
            elif isinstance(artifact, Adata):
                artifact.adata.write(artifact.adata_path)
            elif isinstance(artifact, Figure):
                artifact.fig.savefig(artifact.fig_path, bbox_inches="tight")
            else:
                raise ValueError(f"Artifact {artifact} not recognized.")


def export_artifacts(artifacts: Artifacts):
    """
    Export artifacts to disk.
    """


@dataclasses.dataclass
class Model:
    """
    Model class for storing models.
    """

    model: torch.nn.Module
    model_path: str
    model_name: str


@dataclasses.dataclass
class adata:
    """
    Adata class for storing adatas.
    """

    adata: ad.AnnData
    adata_path: str
    adata_name: str


def export_models(models, model_path):
    """
    Export model to disk.
    """
    for model in models:
        torch.save(model.state_dict(), model_path)


def export_ouput_adatas(adata, fname, out_data_path):
    """
    Export adata to disk.
    """
    adata.write(out_data_path / fname)


def export_figs(fig, fname, fig_dir):
    """
    Export figure to disk.
    """
    fig.savefig(fig_dir / fname, bbox_inches="tight")


# def make_expression_artifacat(adata,model, out_data_path):
#     train_ad = ad.read_h5ad(out_data_path / train_filen.name.replace(H5, OUT + H5))
#     exp_train_ad = make_scvi_normalized_adata(scvi_query, train_ad)

#     # In[ ]:
#     export_ouput_adata(exp_train_ad, train_filen.name.replace(RAW, EXPR), out_data_path)
#     del exp_train_ad, train_ad

#     test_ad = ad.read_h5ad(out_data_path / test_filen.name.replace(H5, OUT + H5))

#     # In[ ]:
#     # reset the cell_type_key before exporting
#     test_ad.obs[cell_type_key] = test_ad.obs["ground_truth"]
#     exp_test_ad = make_scvi_normalized_adata(scvi_query, test_ad)
#     export_ouput_adata(exp_test_ad, test_filen.name.replace(RAW, EXPR), out_data_path)
