import scanpy as sc
from ._mde import mde
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np


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
    frameon = kwargs.pop("frameon", False)
    wspace = kwargs.pop("wspace", 0.35)
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

    sc.pl.embedding(
        adata, basis=basis, color=color, frameon=frameon, wspace=wspace, **kwargs
    )


def plot_predictions(
    adata, pred_key="pred", cell_type_key="cell_type", model_name="LBL8R", title_str=""
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


def plot_scvi_training(
    model_history: dict,
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
    train_elbo = model_history["elbo_train"][1:]
    val_elbo = model_history["elbo_validation"]

    ax = train_elbo.plot()
    val_elbo.plot(ax=ax)

    train_kll = model_history["kl_local_train"][1:]
    train_klg = model_history["kl_global_train"][1:]
    val_kll = model_history["kl_local_validation"]
    val_klg = model_history["kl_global_validation"]

    ax = train_kll.plot()
    val_kll.plot(ax=ax)

    train_loss = model_history["reconstruction_loss_train"][1:]
    val_loss = model_history["reconstruction_loss_validation"]

    ax = train_loss.plot()
    val_loss.plot(ax=ax)


def plot_scanvi_training(
    model_history: dict,
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
    train_elbo = model_history["elbo_train"][1:]
    val_elbo = model_history["elbo_validation"]

    ax = train_elbo.plot()
    val_elbo.plot(ax=ax)

    train_kll = model_history["kl_local_train"][1:]
    train_klg = model_history["kl_global_train"][1:]
    val_kll = model_history["kl_local_validation"]
    val_klg = model_history["kl_global_validation"]

    ax = train_kll.plot()
    val_kll.plot(ax=ax)

    train_loss = model_history["reconstruction_loss_train"][1:]
    val_loss = model_history["reconstruction_loss_validation"]

    ax = train_loss.plot()
    val_loss.plot(ax=ax)

    train_elbo = model_history["elbo_train"][1:]
    train_class = model_history["train_classification_loss"][1:]
    train_f1 = model_history["train_f1_score"][1:]

    ax = train_elbo.plot()
    # val_elbo = model_history["elbo_validation"][1:]
    # val_elbo.plot(ax=ax)
    _ = train_class.plot()
    _ = train_f1.plot()


def plot_lbl8r_training(
    model_history: dict,
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
    train_loss = model_history["train_loss_epoch"][1:]
    validation_loss = model_history["validation_loss"]

    ax = train_loss.plot()
    validation_loss.plot(ax=ax)
