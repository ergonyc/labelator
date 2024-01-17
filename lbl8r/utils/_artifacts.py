## functions for generating and archiving artifacts: figures, models, adatas

import dataclasses
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from ..model.utils._data import Adata
from ..model.utils._Model import Model

from ..model.utils._plot import Figure


@dataclasses.dataclass
class Artifact:
    """
    Artifacts class for storing artifacts.
    """

    artifact: Model | Adata | Figure
    path: str | Path
    # type: str
    name: str

    def __init__(self, artifact, path, name):
        self.artifact = artifact
        self.path = path
        # self.type = type
        self.name = name


@dataclasses.dataclass
class Artifacts:
    """
    Artifacts class for storing artifacts.
    """

    artifacts: list(Artifact)

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
