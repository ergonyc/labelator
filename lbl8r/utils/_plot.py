import scanpy as sc
from ._mde import mde
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np

def plot_embedding(adata: AnnData, basis: str = "X_mde", color: list = None, **kwargs):
    """Plot embedding with `sc.pl.embedding`."""
    # default kwargs
    frameon = kwargs.pop("frameon", False)
    wspace = kwargs.pop("wspace", 0.35)
    device = kwargs.pop("device", "cpu")

    if "X_pca" not in adata.obsm.keys():
       sc.pp.pca(adata)

    if basis == "X_mde" and "X_mde" not in adata.obsm.keys():
        adata.obsm["X_mde"] = mde(adata.obsm["X_pca"], device=device)

    if basis == "X_umap" and "X_umap" not in adata.obsm.keys():
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    if basis == "X_scVI" and "X_scVI" not in adata.obsm.keys():
        scvi_model = kwargs.pop("scvi_model", None)
        if scvi_model is None:
            raise ValueError("Must pass scvi_model to plot scVI embedding")
        scvi_model.setup_anndata(adata)
        X_scVI = scvi_model.get_latent_representation(adata)
        adata.obsm["X_scVI"] = X_scVI[:, :2]

    sc.pl.embedding(adata, basis=basis, color=color, frameon=frameon, wspace=wspace, **kwargs)
    

def plot_predictions(adata, pred_key="pred", cell_type_key="cell_type", model_name="LBL8R", title_str=""):
    """Plot confusion matrix of predictions."""

    df = adata.obs.groupby([pred_key, cell_type_key]).size().unstack(fill_value=0)
    norm_df = df / df.sum(axis=0)

    plt.figure(figsize=(8, 8))
    _ = plt.pcolor(norm_df)
    _ = plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90)
    _ = plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xlabel(f"Predicted ({pred_key})")
    plt.ylabel(f"Observed ({cell_type_key})")
    plt.title(f"{model_name} {title_str} accuracy: {np.mean(adata.obs[pred_key] == adata.obs[cell_type_key]):.3f}")
    plt.colorbar()