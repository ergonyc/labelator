import scanpy as sc
from anndata import AnnData
import numpy as np
from pandas import crosstab as pd_crosstab
from sklearn.metrics import precision_score, recall_score, f1_score

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from ._mde import mde

# -------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------


def savefig_or_show(
    show: bool|None = None,
    save: bool|str = False,
    fig_dir: str|None = None,
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

    if isinstance(save, Path):
        fig_dir += save.root
        filen = save.name

    elif isinstance(save, str):
        # check whether `save` contains a figure extension
        filen = save
        if ext is None:
            ext = "png" # default to png
            for try_ext in ['.svg', '.pdf', '.png']:
                if save.endswith(try_ext):
                    ext = try_ext[1:]
                    filen = filen.replace(try_ext, '')
                    break
        plt.suptitle(save)
        save = True
    else:
        ValueError(f"WTF.. how did we get here save must be a Path or a str, not {type(save)}")

    if save:
        if not Path(fig_dir).exists():
            Path(fig_dir).mkdir(parents=True)

        filename = f"{fig_dir}/{filen}.{ext}"
        print(f"saving figure to file {filename}")
        plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    if save:
        plt.close()  # clear figure

def _prep_save_dir(save,fig_dir,f_prefix):
    """
    force save to be a string and handle fig_dir
    """
    if isinstance(save, str):        
        print(f"found {save}")
    elif isinstance(save, Path):
        if fig_dir is None:
            fig_dir = save.parent
        save = save.name
    elif save:
        save = f"{f_prefix}"
        print(f"converted `save` to `{save}`")

        if isinstance(fig_dir,Path):
            fig_dir = str(fig_dir)
            print(f"{fig_dir=}")
    else:
        fig_dir = None
    return save, fig_dir



def plot_embedding(adata: AnnData, 
        basis: str = "X_mde", 
        color: list = None,
        **kwargs):

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
    device = kwargs.pop("device", "cpu")
    scvi_model = kwargs.pop("scvi_model", None)

    if "X_pca" not in adata.obsm.keys() and basis != "X_scVI":
        print("Computing PCA")
        sc.pp.pca(adata)

    if basis == "X_mde" and "X_mde" not in adata.obsm.keys():
        adata.obsm["X_mde"] = mde(adata.obsm["X_pca"], device=device)

    if basis == "X_umap" and "X_umap" not in adata.obsm.keys():
        print("getting neighbor graph")
        sc.pp.neighbors(adata)
        print("getting umap")
        sc.tl.umap(adata)

    if basis in ["X_scVI", "X_scVI_mde"] and "X_scVI" not in adata.obsm.keys():
        if scvi_model is None:
            raise ValueError("Must pass scvi_model to plot scVI embedding")
        print("getting scvi embedding")

        scvi_model.setup_anndata(adata)
        X_scVI = scvi_model.get_latent_representation(adata)
        adata.obsm["X_scVI"] = X_scVI  # [:, :2]

    if basis == "X_scVI_mde" and "X_scVI_mde" not in adata.obsm.keys():
        adata.obsm["X_scVI_mde"] = mde(adata.obsm["X_scVI"], device=device)

    # force defaults
    frameon = kwargs.pop("frameon", False)
    wspace = kwargs.pop("wspace", 0.35)
    save = kwargs.pop("save", False)
    show = kwargs.pop("show", True)
    fig_dir = kwargs.pop("fig_dir", None)

    # process fig_dir / save
    if isinstance(save, bool): 
        save = "embeddings.png"
    elif isinstance(save, Path):
        if fig_dir is None:
            fig_dir = save.parent

    if fig_dir is not None:  
        print(fig_dir) 
        # HACK: flatten path
        fig_dir = str(fig_dir).replace("/","_")    
        save = f"{fig_dir}_{save}"
        if not Path(fig_dir).exists():
            Path(fig_dir).mkdir(parents=True)

    kwargs.update({"frameon": frameon, 
                "wspace": wspace,
                "show":  show,
                "save": save,
                })


    sc.pl.embedding(
        adata, basis=basis, color=color,**kwargs
    )


def _plot_predictions(
    adata, pred_key="pred", 
    cell_type_key="cell_type", 
    model_name="LBL8R", 
    title_str="", 
    save: bool | Path | str = False,
    show: bool = True, 
    fig_dir: Path|str|None = None,
):
    """Plot confusion matrix of predictions.

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
    

    df = adata.obs.groupby([pred_key, cell_type_key]).size().unstack(fill_value=0)
    norm_df = df / df.sum(axis=0)

    plt.figure(figsize=(8, 8))
    _ = plt.pcolor(norm_df)
    _ = plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90)
    _ = plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xlabel(f"Predicted ({pred_key})")
    plt.ylabel(f"Observed ({cell_type_key})")
    plt.title(
        f"{title_str} accuracy: {np.mean(adata.obs[pred_key] == adata.obs[cell_type_key]):.3f}\n{model_name}"
    )
    plt.colorbar()

    if isinstance(save, str):        
        pass
        # save = f"{model_name}_predictions.png"
    elif isinstance(save, Path):
        if fig_dir is None:
            fig_dir = save.parent
    elif save:
        save = f"{model_name}_predictions.png"
    savefig_or_show(show,save,fig_dir)


def plot_predictions(
    adata, pred_key="pred", 
    cell_type_key="cell_type", 
    model_name="LBL8R", 
    title_str="", 
    save: bool | Path | str = False,
    show: bool = True, 
    fig_dir: Path|str|None = None,
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
    # Calculate precision, recall, and F1-score
    prec = precision_score(df[cell_type_key],  df[pred_key], average='macro')
    rec = recall_score(df[cell_type_key],  df[pred_key], average='macro')
    f1 = f1_score(df[cell_type_key],  df[pred_key], average='macro')
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
    title_str=f"{title_str}: {acc=:3f}:  {prec=:3f}: {rec=:3f}: {f1=:3f}:({model_name})"

    ax.set_title(title_str.split(":")) 

    if isinstance(save, str):        
        pass
        # save = f"{model_name}_predictions.png"
    elif isinstance(save, Path):
        if fig_dir is None:
            fig_dir = save.parent
    elif save:
        save = f"{model_name}_predictions.png"
    savefig_or_show(show,save,fig_dir)


def plot_scvi_training(
    model_history: dict,
    save: bool | Path | str = False,
    show: bool = True, 
    fig_dir: Path|str|None = None,
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
    save, fig_dir = _prep_save_dir(save,fig_dir,"scvi_")
    
    train_elbo = model_history["elbo_train"][1:]
    val_elbo = model_history["elbo_validation"]
    ax = train_elbo.plot()
    val_elbo.plot(ax=ax)
    save_ = save + "elbo" + ".png"
    savefig_or_show(show,save_,fig_dir)

    train_kll = model_history["kl_local_train"][1:]
    val_kll = model_history["kl_local_validation"]
    ax = train_kll.plot()
    val_kll.plot(ax=ax)
    save_ = save + "kl_div" + ".png"
    savefig_or_show(show,save_,fig_dir)

    train_loss = model_history["reconstruction_loss_train"][1:]
    val_loss = model_history["reconstruction_loss_validation"]
    ax = train_loss.plot()
    val_loss.plot(ax=ax)
    save_ = save + "reconstruction_loss" + ".png"
    savefig_or_show(show,save_,fig_dir)


def plot_scanvi_training(
    model_history: dict,
    save: bool | Path | str = False,
    show: bool = True, 
    fig_dir: Path|str|None = None,
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
    save,fig_dir = _prep_save_dir(save,fig_dir,"scanvi_")

    plot_scvi_training(
            model_history,
            save=save,
            show=show,
            fig_dir=fig_dir
            )

    train_class = model_history["train_classification_loss"][1:]
    _ = train_class.plot()
    save_ = save + "reconstruction_loss" + ".png"
    savefig_or_show(show,save_,fig_dir)

    train_f1 = model_history["train_f1_score"][1:]
    _ = train_f1.plot()
    save_ = save + "f1" + ".png"
    savefig_or_show(show,save_,fig_dir)

def plot_lbl8r_training(
    model_history: dict,
    save: bool | Path | str = False,
    show: bool = True, 
    fig_dir: Path|str|None = None,
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

    save,fig_dir = _prep_save_dir(save,fig_dir,"lbl8r_")

    train_loss = model_history["train_loss_epoch"][1:]
    validation_loss = model_history["validation_loss"]
    ax = train_loss.plot()
    validation_loss.plot(ax=ax)
    save_ = save + "train_loss" + ".png"
    savefig_or_show(show,save_,fig_dir)


