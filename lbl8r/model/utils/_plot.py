import scanpy as sc
from anndata import AnnData
import numpy as np
from pandas import crosstab as pd_crosstab
from sklearn.metrics import precision_score, recall_score, f1_score
from dataclasses import dataclass

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scvi.model.base import BaseModelClass as SCVIModel

# from scvi.model import SCVI, SCANVI

from ._mde import mde

from ..._constants import *
from ._data import Adata
from ._lazy_model import LazyModel

"""
logic:  dataclass holds the list of models / names

    iterating through all the models  we can generate the plots for everything at once.

"""


@dataclass
class Figure:
    """
    Figure class for storing figures.
    """

    fig: plt.Figure
    path: str | Path
    name: str
    ext: str
    _closed: bool = False

    #  from enum import Enum?
    # "png" | "svg" | "pdf"
    def __init__(self, fig, path, name, ext: str = "png"):
        self.fig = fig
        self.path = path
        self.name = name
        self.ext = ext

    def savefig(self, show: bool = False):
        """
        Save figure to disk.
        """
        if not Path(self.path).exists():
            Path(self.path).mkdir(parents=True)

        filename = f"{self.path}/{self.name}.{self.ext}"
        print(f"Saving figure to {filename}")
        self.fig.savefig(filename, bbox_inches="tight")

    def show(self):
        """
        Show figure.
        """
        # in case
        if self.fig is not None and self._closed:
            self.fig.show()

    def close(self):
        """
        Close figure.
        """
        self.fig.close()
        self.closed = True

    @property
    def closed(self):
        return self._closed

    @closed.setter
    def closed(self, value: bool):
        self._closed = value


# -------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------
# TODO:  simplify the logic here... pack_fig / prep_save_dir is toooo complicated
def pack_fig(
    fig: plt.Figure,
    file_nm: str,
    show: bool | None = None,
    fig_dir: str | None = None,
    ext: str = None,
):
    """Save current figure to file and/or show it. Adds a suptitle and returns figure and/or shows it

    Parameters
    ----------
    show : bool
        Whether to show the figure. Default is `True`.
    save : bool | Path | str
        If `True` or a `Path` or a `str`, save the figure. Default is `False`.
    fig_dir : Path | str
        Directory to save figure to. Default is `None`.
    ext : str
        Figure extension. Default is `None`.
    """

    if fig_dir is None:
        fig_dir = "."

    if ext is None:
        ext = "png"  # default to png
        for try_ext in [".svg", ".pdf", ".png"]:
            if file_nm.endswith(try_ext):
                ext = try_ext[1:]
                filen = file_nm.replace(try_ext, "")
                break

    if fig_dir is not None:
        fig.suptitle(f"{fig_dir.stem}/{filen}")
    else:
        fig.suptitle(f"...{filen}")

    if show:
        fig.show()

    return Figure(fig, fig_dir, filen, ext)


def plot_embedding(
    adata: AnnData,
    fig_name: str,
    fig_dir: str,
    basis: str = "X_pca",
    color: list = None,
    umap: bool = False,
    **kwargs,
):
    """Plot embedding with `sc.pl.embedding`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    fig_name : str
        Name of figure to save.
    basis : str
        Basis to plot. Default is `X_mde`. Could be: `X_pca`, `X_mde`, `X_umap`, `X_scVI`,`X_scANVI`
    color : list
        List of color keys to plot. Default is `None`.
    **kwargs : dict
        Additional arguments to pass to `sc.pl.embedding`.

    Returns
    -------
    None

    """
    # default kwargs
    device = kwargs.pop("device", None)
    if basis is None:
        print("No basis provided, using PCA")
        basis = PCA_KEY

    if umap:
        if UMAP_KEY not in adata.obsm_keys():
            print("getting neighbor graph")
            sc.pp.neighbors(adata)
            print("getting umap")
            sc.tl.umap(adata)
        basis = UMAP_KEY

    else:  # use mde for visualization

        if basis == PCA_KEY:
            if PCA_KEY not in adata.obsm_keys():
                print("Computing PCA")
                sc.pp.pca(adata)
            if MDE_KEY not in adata.obsm_keys():
                adata.obsm[MDE_KEY] = mde(adata.obsm[PCA_KEY], device=device)
            basis = MDE_KEY

        if basis == SCVI_LATENT_KEY:
            if SCVI_MDE_KEY not in adata.obsm_keys():
                adata.obsm[SCVI_MDE_KEY] = mde(
                    adata.obsm[SCVI_LATENT_KEY], device=device
                )
            basis = SCVI_MDE_KEY

        if basis == SCANVI_LATENT_KEY:
            if SCANVI_MDE_KEY not in adata.obsm_keys():
                adata.obsm[SCANVI_MDE_KEY] = mde(
                    adata.obsm[SCANVI_LATENT_KEY], device=device
                )
            basis = SCANVI_MDE_KEY

    # force defaults
    frameon = kwargs.pop("frameon", False)
    wspace = kwargs.pop("wspace", 0.35)

    fig_name = f"{fig_name}_embeddings"

    kwargs = {
        "frameon": frameon,
        "wspace": wspace,
        "return_fig": True,
    }

    # make sure we have all the color keys... i.e. batch
    clrs = set(color) & set(adata.obs_keys())
    fig = sc.pl.embedding(adata, basis=basis, color=clrs, **kwargs)

    # pack into Figure class
    fig = Figure(fig, fig_dir, fig_name, "png")
    fig.fig.suptitle(f"{fig_name.replace('_','-')} :: {basis}")
    return fig


def plot_predictions(
    adata: AnnData,
    pred_key: str = "pred",
    cell_type_key: str = "cell_type",
    model_name: str = "LBL8R",
    fig_nm: str | None = None,
    show: bool = False,
    fig_dir: Path | str | None = None,
):
    """Plot confusion matrix of predictions. This version is slooooow (6 seconds)

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    pred_key : str
        Key in `adata.obs` where predictions are stored. Default is `pred`.
    cell_type_key : str
        Key in `adata.obs` where cell types are stored. Default is `cell_type`.
    model_name : str
        Name of model. Default is `LBL8R`.
    title_str : str
        Additional string to add to title. Default is `""`.
    fig_dir : Path | str
        Directory to save figure to. Default is `None`.

    Returns
    -------
    None

    """

    df = adata.obs

    # HACK:  this is nasty... but it should work.
    # Keep the first 'pred' and all other columns
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # TODO: fix the problem upstream...

    if df[cell_type_key].isna().sum() > 0:
        print(f"Missing {cell_type_key} labels.  Skipping...")
        print(df[pred_key].value_counts())
        return None

    # Calculate precision, recall, and F1-score
    prec = precision_score(df[cell_type_key], df[pred_key], average="macro")
    rec = recall_score(df[cell_type_key], df[pred_key], average="macro")
    f1 = f1_score(df[cell_type_key], df[pred_key], average="macro")
    acc = (df[pred_key] == df[cell_type_key]).mean()

    confusion_matrix = pd_crosstab(
        df[pred_key],
        df[cell_type_key],
        rownames=[f"Prediction {pred_key}"],
        colnames=[f"Ground truth {cell_type_key}"],
    )
    # TODO: Series.ravel is deprecated. The underlying array is already 1D, so ravel is not necessary.  Use `to_numpy()` for conversion to a numpy array instead.
    confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
    # confusion_matrix /= confusion_matrix.sum(1).to_numpy().reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix,
        cmap=sns.diverging_palette(245, 320, s=60, as_cmap=True),
        ax=ax,
        square=True,
        cbar_kws=dict(shrink=0.4, aspect=12),
    )
    title_str = f"{acc=:3f}:  {prec=:3f}: {rec=:3f}: {f1=:3f})"

    ax.set_title(title_str.split(":"))

    if fig_nm is not None:
        fig_n = f"{fig_nm}_predictions.png"
    else:
        fig_n = f"predictions.png"
    fig = pack_fig(fig, fig_n, fig_dir=fig_dir, show=show)
    return fig


def plot_lbl8r_training(
    model_history: dict,
    fig_nm: str,
    show: bool = True,
    fig_dir: Path | str | None = None,
):
    """Plot training curves of scVI model.

    Parameters
    ----------
    model_history : dict
        History of scVI model.
    fig_nm : str
        Name of figure to save.
    show : bool
        Whether to show the figure. Default is `True`.
    fig_dir : Path | str
        Directory to save figure to. Default is `None`.

    Returns
    -------
    None

    """

    train_loss = model_history["train_loss_epoch"][1:]
    validation_loss = model_history["validation_loss"]
    ax = train_loss.plot()
    validation_loss.plot(ax=ax)
    fig_n = f"{fig_nm}_train_loss.png"
    fig = pack_fig(ax.get_figure(), fig_n, fig_dir=fig_dir, show=show)
    return [fig]


def plot_scvi_training(
    model_history: dict,
    fig_nm: str | None = None,
    show: bool = True,
    fig_dir: Path | str | None = None,
):
    """Plot training curves of scVI model.

    Parameters
    ----------
    model_history : dict
        History of scVI model.
    fig_nm : str | None
        Name of figure to save.
    show : bool
        Whether to show the figure. Default is `True`.
    fig_dir : Path | str
        Directory to save figure to. Default is `None`.

    Returns
    -------
    None

    """
    if fig_nm is None:
        fig_nm = "scvi_"

    figs = []

    train_elbo = model_history["elbo_train"][1:]
    val_elbo = model_history["elbo_validation"]
    ax = train_elbo.plot()
    val_elbo.plot(ax=ax)
    fig_n = f"{fig_nm}_elbo.png"
    fg = pack_fig(ax.get_figure(), fig_n, fig_dir=fig_dir, show=show)
    figs.append(fg)

    train_kll = model_history["kl_local_train"][1:]
    val_kll = model_history["kl_local_validation"]
    ax = train_kll.plot()
    val_kll.plot(ax=ax)
    fig_n = f"{fig_nm}_kl_div.png"
    fg = pack_fig(ax.get_figure(), fig_n, fig_dir=fig_dir, show=show)
    figs.append(fg)

    train_loss = model_history["reconstruction_loss_train"][1:]
    val_loss = model_history["reconstruction_loss_validation"]
    ax = train_loss.plot()
    val_loss.plot(ax=ax)
    fig_n = f"{fig_nm}_reconstruction_loss.png"
    fg = pack_fig(ax.get_figure(), fig_n, fig_dir=fig_dir, show=show)
    figs.append(fg)
    return figs


def plot_scanvi_training(
    model_history: dict,
    fig_nm: str | None = None,
    show: bool = False,
    fig_dir: Path | str | None = None,
):
    """Plot training curves of scVI model.

    Parameters
    ----------
    model_history : dict
        History of scVI model.
    fig_nm : str | None
        Name of figure to save.
    show : bool
        Whether to show the figure. Default is `True`.
    fig_dir : Path | str
        Directory to save figure to. Default is `None`.

    Returns
    -------
    None


    """
    if fig_nm is None:
        fig_nm = "scanvi_"

    figs = plot_scvi_training(model_history, fig_nm=fig_nm, show=show, fig_dir=fig_dir)
    figs = []

    if "train_classification_loss" not in model_history.keys():
        return figs

    train_class = model_history["train_classification_loss"][1:]
    ax = train_class.plot()  # is dumping the return teh right thing to do?
    fig_n = f"{fig_nm}_reconstruction_loss.png"
    fg = pack_fig(ax.get_figure(), fig_n, fig_dir=fig_dir, show=show)
    figs.append(fg)

    train_f1 = model_history["train_f1_score"][1:]
    ax = train_f1.plot()
    fig_n = f"{fig_nm}_f1.png"
    fg = pack_fig(ax.get_figure(), fig_n, fig_dir=fig_dir, show=show)
    figs.append(fg)

    return figs
