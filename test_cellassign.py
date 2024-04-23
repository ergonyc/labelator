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
# raw_filen = data_path / XYLENA2_RAW_ANNDATA
# raw_ad = ad.read_h5ad(raw_filen)


# In[ ]:
#####################

ns_top_genes = [20_000, 10_000, 5_000, 3_000, 2_000, 1_000]
ds_names = ["20k", "10k", "5k", "3k", "2k", "1k"]


# In[ ]:

cell_types = {}
# In[ ]:

# #  1. load marker_genes
# markers = pd.read_csv("celltype_marker_table.csv", index_col=0)
# # defensive
# markers = markers[~markers.index.duplicated(keep="first")].rename_axis(index=None)


# In[ ]:


def get_cell_types(adata, markers):

    #  2. copy for cellassign
    # bdata = adata[:, markers.index].copy() #
    bdata = adata[:, adata.var.index.isin(markers.index)].copy()

    #  3. get size_factor and noise
    lib_size = bdata.X.sum(1)  # type: ignore
    bdata.obs["size_factor"] = lib_size / np.mean(lib_size)

    print("size_factor", bdata.obs["size_factor"].mean())
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
    preds = model.predict()

    bdata.obs["cellassign_types"] = preds.idxmax(axis=1).values

    # 7. transfer cell_type to adata
    adata.obs["cellassign_types"] = bdata.obs["cellassign_types"]

    #  8. save model & artificts
    predictions = (
        bdata.obs[["sample", "cellassign_types", "cell_type"]]
        .reset_index()
        .rename(columns={"index": "cells"})
    )

    preds["sample"] = predictions["sample"]
    preds["cellassign_types"] = predictions["cellassign_types"]
    preds["cell_type"] = predictions["cell_type"]
    preds["cells"] = predictions["cells"]
    preds.index = preds["cells"]

    return preds, model


# In[ ]:
# taxonomy from @nick

neuron_subs = dict(
    glutamatergic=["SLC17A6", "NEUROD6", "SATB2"],
    gabergic=["SLC32A1", "GAD2"],
    dopaminergic=["SLC6A3", "SLC18A2"],
)
astro_sub = dict(protoplasmic=["GJA1"], fibrous=["GFAP"])
immune_sub = dict(microglia=["P2RY12"], lymphoid=["SKAP1"])
lymphoid_sub = dict(t_cells=["CD8B", "CD8A"], b_cells=["IGHG1"])

neurons = ["GRIN2A", "RBFOX3"]
glutamatergic = neurons + neuron_subs["glutamatergic"]
gabergic = neurons + neuron_subs["gabergic"]
dopaminergic = neurons + neuron_subs["dopaminergic"]

astrocytes = ["AQP4"]
astrocyte_proto = astrocytes + astro_sub["protoplasmic"]
astrocyte_fibro = astrocytes + astro_sub["fibrous"]

oligos = ["CLDN11", "CNP", "PLP1", "ST18"]
OPCs = ["LHFPL3", "PDGFRA"]
choroid = ["TTR", "KRT18", "FOLR1"]

immune_cells = ["PTPRC"]
microglia = immune_cells + immune_sub["microglia"]

lymphoid = immune_cells + immune_sub["lymphoid"]
t_cells = lymphoid + lymphoid_sub["t_cells"]
b_cells = lymphoid + lymphoid_sub["b_cells"]


cell_types = [
    "neurons",
    "glutamatergic",
    "gabergic",
    "dopaminergic",
    "astrocytes",
    "astrocyte_proto",
    "astrocyte_fibro",
    "oligos",
    "OPCs",
    "choroid",
    "immune_cells",
    "microglia",
    "lymphoid",
    "t_cells",
    "b_cells",
]

# In[ ]:

colnms = []
colnms = [eval(ct) for ct in cell_types]
col = []
for e in colnms:
    col += e
# In[ ]:
import numpy as np

marker = np.unique(col)


# In[ ]:
import pandas as pd

df = pd.DataFrame(index=marker)

df["neurons"] = df.index.isin(neurons)
df["glutamatergic"] = df.index.isin(glutamatergic)
df["gabergic"] = df.index.isin(gabergic)
df["dopaminergic"] = df.index.isin(dopaminergic)
df["astrocytes"] = df.index.isin(astrocytes)
df["astrocyte_proto"] = df.index.isin(astrocyte_proto)
df["astrocyte_fibro"] = df.index.isin(astrocyte_fibro)
df["oligos"] = df.index.isin(oligos)
df["OPCs"] = df.index.isin(OPCs)
df["choroid"] = df.index.isin(choroid)
df["immune_cells"] = df.index.isin(immune_cells)
df["microglia"] = df.index.isin(microglia)
df["lymphoid"] = df.index.isin(lymphoid)
df["t_cells"] = df.index.isin(t_cells)
df["b_cells"] = df.index.isin(b_cells)

filen = raw_data_path / "new_taxonomy_table.csv"
df.to_csv(filen)

# In[ ]:
filen = raw_data_path / "new_taxonomy_table.csv"
markers_new = pd.read_csv(filen, index_col=0)


markers_bottom_level = markers_new[
    [
        "glutamatergic",
        "gabergic",
        "dopaminergic",
        "astrocyte_proto",
        "astrocyte_fibro",
        "oligos",
        "OPCs",
        "choroid",
        "microglia",
        "t_cells",
        "b_cells",
    ]
]

# In[ ]:

# In[ ]:
markers = markers_bottom_level
filen = raw_data_path / "celltype_marker_table2.csv"
markers.to_csv(filen)
# filen = data_path / XYLENA2_TRAIN
# adata = ad.read_h5ad(filen)
# train_predictions, train_model = get_cell_types(adata, markers)
# train_predictions.to_csv(
#     "train_labels.csv", index=False
# )  # # pred_file = "cellassign_predictions.csv"

# # In[ ]:
# XYLENA2_FULL = "xyl2_full.h5ad"
# filen = data_path / XYLENA2_FULL
# adata = ad.read_h5ad(filen)

# tmp2 = probe_model(adata, train_model, markers)


# # In[ ]:
# filen = data_path / XYLENA2_TEST
# adata = ad.read_h5ad(filen)
# test_predictions, test_model = get_cell_types(adata, markers)
# test_predictions.to_csv(
#     "test_labels.csv", index=False
# )  # # pred_file = "cellassign_predictions.csv"

# # In[ ]:


# filen = data_path / XYLENA2_QUERY
# adata = ad.read_h5ad(filen)
# query_predictions, query_model = get_cell_types(adata, markers)
# query_predictions.to_csv(
#     "query_labels.csv", index=False
# )  # # pred_file = "cellassign_predictions.csv"
# In[ ]:

XYLENA2_FULL = "xyl2_full.h5ad"
filen = data_path / XYLENA2_FULL
adata = ad.read_h5ad(filen)

# In[ ]:

full_predictions, full_model = get_cell_types(adata, markers)
# In[ ]:

filen = data_path / "full_labels.csv"

full_predictions.to_csv(filen, index=False)


# In[ ]:

filen = data_path / "full_cellassign.pkl"

full_model.save(filen)


# In[ ]:

# In[ ]:


# In[ ]:
train_predictions = pd.read_csv("train_labels.csv")
full_predictions = pd.read_csv("full_labels.csv")
# find the same cells in full and train

idx2 = full_predictions[full_predictions["cells"].isin(train_predictions["cells"])]
train_predictions["cells2"] = idx2["cellassign_types"].values
train_predictions.head()


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

    # df = adata.obs

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


# In[ ]:

plot_predictions(train_predictions, pred_key="cellassign_types", cell_type_key="cells2")
# In[ ]:
plot_predictions(train_predictions)
# %%
plot_predictions(test_predictions)
# %%
plot_predictions(query_predictions)
# %%
plot_predictions(full_predictions)

# %%
