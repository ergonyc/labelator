#!/usr/bin/env python
# coding: utf-8

### set up train and test AnnData objects for LBL8R

# In[ ]:
### import local python functions in ../lbl8r
import sys
import os

import scanpy as sc
import anndata as ad
from pathlib import Path
import scipy.sparse as sp
import pandas as pd
import scvi
import pickle
import numpy as np

sys.path.append(os.path.abspath("/media/ergonyc/Projects/SingleCell/labelator/"))

from lbl8r.model.utils._data import transfer_pcs

from lbl8r._constants import *


# ## create train and query datasets.

## e.g. for xylena data
XYLENA2_RAW_ANNDATA = "full_object.h5ad"
XYLENA2_GROUND_TRUTH = "cellassign_predictions.csv"

XYLENA2_FULL = "xyl2_full.h5ad"

XYLENA2_TRAIN = "xyl2_train.h5ad"
XYLENA2_TEST = "xyl2_test.h5ad"
XYLENA2_QUERY = "xyl2_query.h5ad"

XYLENA2_RAW_PATH = "data/scdata/xylena_raw"
XYLENA2_PATH = "data/scdata/xylena"


## load raw data.
# In[ ]:
root_path = Path.cwd()
data_path = root_path / XYLENA2_PATH
raw_data_path = root_path / XYLENA2_RAW_PATH


# In[ ]: Load raw data
# raw_filen = raw_data_path / XYLENA2_RAW_ANNDATA
# raw_ad = ad.read_h5ad(raw_filen)


# In[ ]:
#####################

ns_top_genes = [20_000, 10_000, 5_000, 3_000, 1_000]
ds_names = ["20k", "10k", "5k", "3k", "1k"]

ns_top_genes = [5_000, 3_000, 1_000]
ds_names = ["5k", "3k", "1k"]

# In[ ]:

cell_types = {}
# In[ ]:


#  1. load marker_genes
markers = pd.read_csv("celltype_marker_table.csv", index_col=0)
# defensive
markers = markers[~markers.index.duplicated(keep="first")].rename_axis(index=None)


# genes = {}
# ds_path = root_path / f"{XYLENA2_PATH}{'10k'}"

# train_filen = ds_path / XYLENA2_TRAIN
# adata = ad.read_h5ad(train_filen)


def get_cell_types(adata, markers):

    #  2. copy for cellassign
    # bdata = adata[:, markers.index].copy() #
    bdata = adata[:, adata.var.index.isin(markers.index)].copy()

    #  3. get size_factor and noise
    lib_size = bdata.X.sum(1)  # type: ignore
    bdata.obs["size_factor"] = lib_size / np.mean(lib_size)

    #  4. model = CellAssign(bdata, marker_genes)
    scvi.external.CellAssign.setup_anndata(
        bdata,
        size_factor_key="size_factor",
        batch_key="sample",
        layer=None,  #'counts',
        # continuous_covariate_keys=noise
    )

    #  5. model.train()
    model = scvi.external.CellAssign(bdata, markers)
    plan_args = {"lr_factor": 0.05, "lr_patience": 20, "reduce_lr_on_plateau": True}
    model.train(
        max_epochs=1000,
        accelerator="gpu",
        early_stopping=True,
        plan_kwargs=plan_args,
        early_stopping_patience=40,
    )

    #  6. model.predict()
    bdata.obs["cellassign_types"] = model.predict().idxmax(axis=1).values

    # 7. transfer cell_type to adata
    adata.obs["cellassign_types"] = bdata.obs["cellassign_types"]

    #  8. save model & artificts
    predictions = (
        bdata.obs[["sample", "cellassign_types", "cell_type"]]
        .reset_index()
        .rename(columns={"index": "cells"})
    )
    predictions.to_csv(
        "train_labels", index=False
    )  # # pred_file = "cellassign_predictions.csv"

    return predictions, model


# In[ ]:

filen = raw_data_path / XYLENA2_TRAIN
adata = ad.read_h5ad(filen)
train_predictions, train_model = get_cell_types(adata, markers)
# In[ ]:


filen = raw_data_path / XYLENA2_TEST
adata = ad.read_h5ad(filen)
test_predictions, test_model = get_cell_types(adata, markers)
# In[ ]:


filen = raw_data_path / XYLENA2_QUERY
adata = ad.read_h5ad(filen)
query_predictions, query_model = get_cell_types(adata, markers)
# In[ ]:

XYLENA2_FULL = "xyl2_full.h5ad"
filen = raw_data_path / XYLENA2_FULL
adata = ad.read_h5ad(filen)
full_predictions, full_model = get_cell_types(adata, markers)


# In[ ]:

from pandas import crosstab as pd_crosstab
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def plot_predictions(
    df: pd.DataFrame,
    pred_key: str = "cellassign_types",
    cell_type_key: str = "cell_type",
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
    title_str = f"{acc=:3f}:  {prec=:3f}: {rec=:3f}: {f1=:3f})"

    ax.set_title(title_str.split(":"))


# In[ ]:
plot_predictions(train_predictions)
# %%
plot_predictions(test_predictions)
# %%
plot_predictions(query_predictions)
# %%
plot_predictions(full_predictions)

# %%
