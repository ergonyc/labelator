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
        self.fig.savefig(filename, bbox_inches="tight")
        self.show(False)  # clear figure
        # self.fig.savefig(self.fig_path, bbox_inches="tight")

    def show(self, show: bool = True):
        """
        Show figure.
        """
        if show:
            self.fig.show()
        # else:
        #     self.fig.close()


# -------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------
# TODO:  simplify the logic here... savefig_or_show / prep_save_dir is toooo complicated
def savefig_or_show(
    show: bool | None = None,
    save: bool | str = False,
    fig_dir: str | None = None,
    ext: str = None,
):
    """Save current figure to file and/or show it.

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

    elif isinstance(save, str):
        # check whether `save` contains a figure extension
        filen = save
        if ext is None:
            ext = "png"  # default to png
            for try_ext in [".svg", ".pdf", ".png"]:
                if save.endswith(try_ext):
                    ext = try_ext[1:]
                    filen = filen.replace(try_ext, "")
                    break
        save = True
    else:
        ValueError(
            f"WTF.. how did we get here save must be a Path or a str, not {type(save)}"
        )
    fig = Figure(plt.gcf(), fig_dir, filen, ext)
    fig.fig.suptitle(f"{fig_dir.stem}/{filen}")
    return fig


# def plot_all(
#     adata: AnnData,
#     plots: list = ["embedding", "predictions", "training"],
#     model: SCVIModel = None,
#     model_name: str = "LBL8R",
#     emb_kwargs: dict = None,
#     pred_kwargs: dict = None,
#     fig_kwargs: dict = None,
# ):
#     """Plot all the things.

#     Parameters
#     ----------
#     adata : AnnData
#         Annotated data matrix.
#     model : scvi.models.VAE | scvi.models.SCANVI | None
#         grainable
#     save : bool | Path | str
#         If `True` or a `Path` or a `str`, save the figure. Default is `False`.
#     show : bool
#         Whether to show the figure. Default is `True`.
#     fig_dir : Path | str
#         Directory to save figure to. Default is `None`.

#     Returns
#     -------
#     None

#     """
#     figs = []
#     # fig_kwargs =dict(save=save,show=show,fig_dir=fig_dir)
#     if "embedding" in plots:
#         fg = plot_embedding(
#             adata,
#             **emb_kwargs,
#             **fig_kwargs,
#         )
#         figs.extend(fg)

#     if "predictions" in plots:
#         fg = plot_predictions(
#             adata,
#             **pred_kwargs,
#             **fig_kwargs,
#         )
#         figs.extend(fg)

#     if "training" in plots:
#         if model.__class__.__name__ == "LBL8R":
#             # if isinstance(model, LBL8R):
#             plot_lbl8r_training(model.history, **fig_kwargs)
#         elif model.__class__.__name__ == "SCANVI":
#             # elif isinstance(model, SCANVI):
#             plot_scanvi_training(model.history, **fig_kwargs)
#         elif model.__class__.__name__ == "SCVI":
#             # elif isinstance(model, SCVI):
#             fg = plot_scvi_training(model.history, **fig_kwargs)
#         else:  # xgb?
#             pass
#         figs.extend(fg)
#     return figs


def plot_embedding(adata: AnnData, basis: str = "X_mde", color: list = None, **kwargs):
    """Plot embedding with `sc.pl.embedding`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    basis : str
        Basis to plot. Default is `X_mde`. Could be: `X_pca`, `X_mde`, `X_umap`, `X_scVI`, `X_scVI_mde`.
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
    # TODO: depricate the scvi_model kwarg and scvi expression on the fly
    scvi_model = kwargs.pop("scvi_model", None)

    if "X_pca" not in adata.obsm_keys() and basis != "X_scVI":
        print("Computing PCA")
        sc.pp.pca(adata)

    if basis == "X_mde" and "X_mde" not in adata.obsm_keys():
        adata.obsm["X_mde"] = mde(adata.obsm["X_pca"], device=device)

    if basis == "X_umap" and "X_umap" not in adata.obsm_keys():
        print("getting neighbor graph")
        sc.pp.neighbors(adata)
        print("getting umap")
        sc.tl.umap(adata)

    if basis in ["X_scVI", "X_scVI_mde"] and "X_scVI" not in adata.obsm_keys():
        if scvi_model is None:
            raise ValueError("Must pass scvi_model to plot scVI embedding")
        print("getting scvi embedding")

        scvi_model.setup_anndata(adata)
        X_scVI = scvi_model.get_latent_representation(adata)
        adata.obsm["X_scVI"] = X_scVI  # [:, :2]

    if basis == "X_scVI_mde" and "X_scVI_mde" not in adata.obsm_keys():
        adata.obsm["X_scVI_mde"] = mde(adata.obsm["X_scVI"], device=device)

    # force defaults
    frameon = kwargs.pop("frameon", False)
    wspace = kwargs.pop("wspace", 0.35)
    save = kwargs.pop("save", False)
    show = kwargs.pop("show", False)
    fig_dir = kwargs.pop("fig_dir", None)

    # process fig_dir / save
    if isinstance(save, bool):
        save = f"embeddings"
    else:
        save = f"{save}_embeddings"

    kwargs.update(
        {
            "frameon": frameon,
            "wspace": wspace,
            "return_fig": True,
        }
    )

    fig = sc.pl.embedding(adata, basis=basis, color=color, **kwargs)
    fig = Figure(fig, fig_dir, save)
    return fig


def plot_predictions(
    adata: AnnData,
    pred_key: str = "pred",
    cell_type_key: str = "cell_type",
    model_name: str = "LBL8R",
    title_str: str = "",
    save: bool | str = False,
    show: bool = True,
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
    fig_dir :

    Returns
    -------
    None

    """

    df = adata.obs

    # HACK:  this is nasty... but it should work.
    # Keep the first 'pred' and all other columns
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # TODO: fix the problem upstream...

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
    confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix,
        cmap=sns.diverging_palette(245, 320, s=60, as_cmap=True),
        ax=ax,
        square=True,
        cbar_kws=dict(shrink=0.4, aspect=12),
    )
    title_str = f"{title_str}: {acc=:3f}:  {prec=:3f}: {rec=:3f}: {f1=:3f})"

    ax.set_title(title_str.split(":"))

    if isinstance(save, str):
        save = f"predictions_{save}.png"
    elif save:
        save = f"predictions.png"
    fig = savefig_or_show(show, save, fig_dir)
    return fig


def plot_lbl8r_training(
    model_history: dict,
    save: bool | str = False,
    show: bool = True,
    fig_dir: Path | str | None = None,
):
    """Plot training curves of scVI model.

    Parameters
    ----------
    model_history : dict
        History of scVI model.

    TODO:  wrap things so that empty keys don't break things
    -------
    None

    """

    if isinstance(save, bool):
        save = ""

    train_loss = model_history["train_loss_epoch"][1:]
    validation_loss = model_history["validation_loss"]
    ax = train_loss.plot()
    validation_loss.plot(ax=ax)
    save_ = save + "train_loss" + ".png"
    fig = savefig_or_show(show, save_, fig_dir)
    return [fig]


def plot_scvi_training(
    model_history: dict,
    save: bool | str = False,
    show: bool = True,
    fig_dir: Path | str | None = None,
):
    """Plot training curves of scVI model.

    Parameters
    ----------
    model_history : dict
        History of scVI model.

    TODO:  wrap things so that empty keys don't break things
    -------
    None

    """

    if isinstance(save, bool):
        save = "scvi_"

    figs = []

    train_elbo = model_history["elbo_train"][1:]
    val_elbo = model_history["elbo_validation"]
    ax = train_elbo.plot()
    val_elbo.plot(ax=ax)
    save_ = save + "elbo.png"
    fg = savefig_or_show(show, save_, fig_dir)
    figs.append(fg)

    train_kll = model_history["kl_local_train"][1:]
    val_kll = model_history["kl_local_validation"]
    ax = train_kll.plot()
    val_kll.plot(ax=ax)
    save_ = save + "kl_div.png"
    fg = savefig_or_show(show, save_, fig_dir)
    figs.append(fg)

    train_loss = model_history["reconstruction_loss_train"][1:]
    val_loss = model_history["reconstruction_loss_validation"]
    ax = train_loss.plot()
    val_loss.plot(ax=ax)
    save_ = save + "reconstruction_loss.png"
    fg = savefig_or_show(show, save_, fig_dir)
    figs.append(fg)
    return figs


def plot_scanvi_training(
    model_history: dict,
    save: bool | str = False,
    show: bool = True,
    fig_dir: Path | str | None = None,
):
    """Plot training curves of scVI model.

    Parameters
    ----------
    model_history : dict
        History of scVI model.

    TODO:  wrap things so that empty keys don't break things
    -------
    None

    """
    if isinstance(save, bool):
        save = "scanvi_"

    figs = plot_scvi_training(model_history, save=save, show=show, fig_dir=fig_dir)
    # figs = []

    train_class = model_history["train_classification_loss"][1:]
    _ = train_class.plot()  # is dumping the return teh right thing to do?
    save_ = save + "reconstruction_loss.png"
    fg = savefig_or_show(show, save_, fig_dir)
    figs.append(fg)

    train_f1 = model_history["train_f1_score"][1:]
    _ = train_f1.plot()
    save_ = save + "f1.png"
    fg = savefig_or_show(show, save_, fig_dir)
    figs.append(fg)

    return figs


def make_plots(
    data: Adata,
    model: LazyModel,
    train_or_query: str,
    labels_key: str,
    path: Path | str | None = None,
) -> list[Figure]:
    """
    make all the plots

    Parameters
    ----------
    data : Adata
        Annotated data matrix.
    model : LazyModel
        LazyModel object.
    test_or_query : str
        Whether we are testing or querying.
    labels_key : str
        Key in `adata.obs` where cell types are stored. Default is `cell_type`.

    Returns
    -------
    list[Figure]
        List of figures.
    """

    if gen_plots := data is not None:
        ad = data.adata
    else:
        print(f"WARNING!!! not data provided, only generating training plots")

    fig_dir = path / "figs" / model.name
    title_str = f"{train_or_query.upper()}-{model.name}"

    basis = SCVI_MDE_KEY if model.name.endswith(EMB) else MDE_KEY

    fig_kwargs = dict(fig_dir=fig_dir, save=train_or_query, show=False)
    figs = []

    if gen_plots:
        # PLOT embeddings ###############################################################
        fg = plot_predictions(
            ad,
            pred_key="pred",
            cell_type_key=labels_key,
            model_name=model.name,
            title_str=title_str,
            **fig_kwargs,
        )
        figs.append(fg)
        # PLOT embeddings ###############################################################
        fg = plot_embedding(
            ad,
            basis=basis,
            color=[labels_key, "batch"],
            **fig_kwargs,
        )
        fg.fig.suptitle(f"{title_str} :: {basis}")
        figs.append(fg)
        # update with the embedding
        data.update(ad)

    # PLOT TRAINING ###############################################################
    if model.name.startswith("lbl8r_") and train_or_query == "train":
        fg = plot_lbl8r_training(model.model.history, **fig_kwargs)
        figs.extend(fg)  # training returns a list of figures

    # BUG: we need to select the sub-models differently now. Need to either pass tehe model_set or call this repeatedly...
    elif model.name.startswith("scanvi"):
        if train_or_query == "train":
            # # plot scvi, scanvi, qscvi, and qscanvi (model)
            # fg = plot_scvi_training(model.vae.history, **fig_kwargs)
            # figs.extend(fg)
            # fg = plot_scanvi_training(model.scanvi.history, **fig_kwargs)
            # figs.extend(fg)
            pass
        else:
            # fg = plot_scvi_training(
            #     model.q_vae.history, fig_dir=fig_dir, save="query_scvi_", show=False
            # )
            # figs.extend(fg)
            fg = plot_scvi_training(
                model.model.history, fig_dir=fig_dir, save="query_scanvi_", show=False
            )
            figs.extend(fg)

    return figs
