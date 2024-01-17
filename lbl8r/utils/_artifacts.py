## functions for generating and archiving artifacts: figures, models, adatas

import dataclasses
import matplotlib.pyplot as plt

from ..model.utils._data import Adata
from ..model.utils._Model import Model


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
    pass


def export_figs(fig, fname, fig_dir):
    """
    Export figure to disk.
    """
    fig.savefig(fig_dir / fname, bbox_inches="tight")
