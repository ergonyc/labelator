## functions for generating and archiving artifacts: figures, models, adatas

from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from ..model.utils._data import Adata
from ..model.utils._Model import Model

from ..model.utils._plot import Figure


@dataclass
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

    def export_artifact(self):
        pass


@dataclass
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
                model = artifact.model
                model.export(artifact.model_path, artifact.name)

            elif isinstance(artifact, Adata):
                artifact.adata.write(artifact.adata_path)
            elif isinstance(artifact, Figure):
                artifact.fig.savefig(artifact.fig_path, bbox_inches="tight")
            else:
                raise ValueError(f"Artifact {artifact} not recognized.")
