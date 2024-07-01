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
import numpy as np
import scvi


sys.path.append(os.path.abspath("/media/ergonyc/Projects/SingleCell/labelator/"))

from lbl8r.model.utils._data import transfer_pcs

from lbl8r._constants import *

# In[ ]:
## e.g. for xylena data
XYLENA2_RAW_ANNDATA = "full_object.h5ad"
XYLENA2_GROUND_TRUTH = "ground_truth_labels.csv"

XYLENA2_FULL = "xyl2_full.h5ad"

XYLENA2_TRAIN = "xyl2_train.h5ad"
XYLENA2_TEST = "xyl2_test.h5ad"
XYLENA2_QUERY = "xyl2_query.h5ad"

XYLENA2_RAW_PATH = "data/scdata/xylena_raw"
XYLENA2_PATH = "data/scdata/xylena"

XYLENA2_FULL_HVG = "xyl2_full_hvg.csv"
XYLENA2_TRAIN_HVG = "xyl2_train_hvg.csv"

# XYLENA2_FULL_LABELS = "full_labels.csv"
XYLENA2_FULL_LABELS = "full_labels.feather"
XYLENA2_FULL_CELLASSIGN = "full_cellassign_model"


# XYLENA2_FULL_LABELS = "full_labels.csv"
XYLENA2_TRAIN_LABELS = "train_labels.feather"
XYLENA2_TRAIN_CELLASSIGN = "train_cellassign_model"


## load raw data.

# In[ ]:
root_path = Path.cwd()
data_path = root_path / XYLENA2_PATH
raw_data_path = root_path / XYLENA2_RAW_PATH

# In[ ]:
# taxonomy from @nick

NEURON = ["GRIN2A", "RBFOX3"]
ASTROCYTE = ["AQP4", "RFX4"]
OLIGO = ["CLDN11", "CNP", "PLP1", "ST18", "MBP", "MOG", "MAG"]
OPC = ["LHFPL3", "MEGF11", "PCDH15", "PDGFRA"]
IMMUNE = ["PTPRC"]
BLOOD_VESSEL = ["CD34"]


neuron_subs = dict(
    glutamatergic=["SLC17A6", "NEUROD6", "SATB2"],
    gabergic=["SLC32A1", "GAD2", "LHX6"],
    # dopaminergic=["SLC6A3", "SLC18A2"],
)

astro_sub = dict(protoplasmic=["GJA1"], fibrous=["GFAP", "CD44"])

immune_sub = dict(
    microglia=["P2RY12"], t_cell=["CD8B", "CD8A", "CD3E"], b_cell=["IGHG1"]
)


blood_sub = dict(
    pericytes=["HIGD1B", "ABCC9", "NDUFA4L2", "NOTCH3", "RGS5"],
    endothelial=["PECAM1", "FLT1", "KDR", "SLC2A1", "VWF", "CLDN5", "TIE1"],
)


neuron_other = NEURON
glutamatergic = NEURON + neuron_subs["glutamatergic"]
gabergic = NEURON + neuron_subs["gabergic"]

astrocyte_other = ASTROCYTE
protoplasmic_astrocyte = ASTROCYTE + astro_sub["protoplasmic"]
fibrous_astrocyte = ASTROCYTE + astro_sub["fibrous"]

immune_other = IMMUNE
microglia = IMMUNE + immune_sub["microglia"]
t_cell = IMMUNE + immune_sub["t_cell"]
b_cell = IMMUNE + immune_sub["b_cell"]

blood_other = BLOOD_VESSEL
pericyte = BLOOD_VESSEL + blood_sub["pericytes"]
endothelial = BLOOD_VESSEL + blood_sub["endothelial"]

oligo = OLIGO
opc = OPC
unknown = []

cell_types = [
    "oligo",
    "opc",
    "glutamatergic",
    "gabergic",
    "protoplasmic_astrocyte",
    "fibrous_astrocyte",
    "microglia",
    "t_cell",
    "b_cell",
    "pericyte",
    "endothelial",
    "unknown",
]

# "loose" taxonomy includes a non-subtype for our top-level cell types
# "neuron_other",
# "astrocyte_other",
# "immune_other",
# "blood_other",


# ]


# In[ ]:

colnms = []
colnms = [eval(ct) for ct in cell_types]
col = []
for e in colnms:
    col += e
# In[ ]:
marker = np.unique(col)

# In[ ]:
df = pd.DataFrame(index=marker)

for t in cell_types:
    tt = eval(t)
    df[t] = df.index.isin(tt)

# In[ ]:
df = df.astype(int)
df.to_csv("new_taxonomy_table.csv")

# In[ ]:
markers_new = pd.read_csv("new_taxonomy_table.csv", index_col=0)


# In[ ]:
# In[ ]:
markers = markers_new


# In[ ]:


def get_cell_types(adata, markers, batch_key=None, noise=None):

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
        # batch_key="sample",
        batch_key=batch_key,
        layer=None,  #'counts',
        ontinuous_covariate_keys=noise
    )

    #  5. model.train()
    model = scvi.external.CellAssign(bdata, markers)
    model.train()
    # plan_args = {"lr_factor": 0.05, "lr_patience": 20, "reduce_lr_on_plateau": True}
    # model.train(
    #     max_epochs=1000,
    #     accelerator="gpu",
    #     early_stopping=True,
    #     plan_kwargs=plan_args,
    #     early_stopping_patience=40,
    # )

    #  6. model.predict()
    preds = model.predict()

    bdata.obs["cellassign_types"] = preds.idxmax(axis=1).values

    # 7. transfer cell_type to adata
    adata.obs["cellassign_types"] = bdata.obs["cellassign_types"]

    if "cell_type" not in bdata.obs.columns:
        bdata.obs["cell_type"] = "NONE"

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


    preds["cellassign_types"] = preds.idxmax(axis=1).values

    # # 7. transfer cell_type to adata
    # adata.obs["cellassign_types"] = preds["cellassign_types"]

    if "cell_type" not in bdata.obs.columns:
        preds["cell_type"] = "NONE"
    else:
        preds["cell_type"] = bdata.obs["cell_type"]

    preds["sample"] = bdata.obs["sample"]
    preds["cells"] = bdata.obs.index.values
    preds.index = preds["cells"]


    return preds, model


# In[ ]: Load raw data

input_data = 

########################
# 0. LOAD TRAIN DATA
########################
train_filen = raw_data_path / XYLENA2_TRAIN  # XYLENA2_RAW_ANNDATA
train_ad = ad.read_h5ad(train_filen)

# In[ ]:
# In[ ]: Loose
noise = ['doublet_score', 'pct_counts_mt', 'pct_counts_rb'] # aka "noise"
train_predictions, train_model = get_cell_types(train_ad, markers, batch_key="sample", noise=noise)

filen = raw_data_path / XYLENA2_TRAIN_LABELS
train_predictions.reset_index(drop=True).to_feather(filen)
# In[ ]:
filen = raw_data_path / XYLENA2_TRAIN_CELLASSIGN
train_model.save(filen, overwrite=True)

train_predictions.cellassign_types.value_counts()


# In[ ]: Load raw data
########################
# 0. LOAD RAW DATA
########################

raw_filen = raw_data_path / XYLENA2_RAW_ANNDATA
raw_ad = ad.read_h5ad(raw_filen)
# In[ ]:

bdata = raw_ad[:, raw_ad.var.index.isin(markers.index)].copy()
#  3. get size_factor and noise
lib_size = bdata.X.sum(1)  # type: ignore
bdata.obs["size_factor"] = lib_size / np.mean(lib_size)
batch_key = "sample"
print("size_factor", bdata.obs["size_factor"].mean())

train_model.setup_anndata(
        bdata,
        size_factor_key="size_factor",
        batch_key=batch_key,
        layer=None, 
    )

preds = train_model.predict()

preds["cellassign_types"] = preds.idxmax(axis=1).values

# # 7. transfer cell_type to adata
# adata.obs["cellassign_types"] = preds["cellassign_types"]

if "cell_type" not in bdata.obs.columns:
    preds["cell_type"] = "NONE"
else:
    preds["cell_type"] = bdata.obs["cell_type"]


# #  8. save model & artificts
# predictions = (
#     bdata.obs[["sample", "cellassign_types", "cell_type"]]
#     .reset_index()
#     .rename(columns={"index": "cells"})
# )
preds["cells"] = bdata.obs.index.values
preds.index = preds["cells"]


# In[ ]: Loose
full_predictions, full_model = get_cell_types(raw_ad, markers, batch_key="sample")

# In[ ]:
filen = raw_data_path / XYLENA2_FULL_LABELS
full_predictions.reset_index(drop=True).to_feather(filen)
# In[ ]:
filen = raw_data_path / XYLENA2_FULL_CELLASSIGN
full_model.save(filen, overwrite=True)

full_predictions.cellassign_types.value_counts()


# In[ ]: summarize


filen = raw_data_path / XYLENA2_TRAIN_LABELS
train_predictions2 = pd.read_feather(filen)

filen = raw_data_path / XYLENA2_FULL_LABELS
full_predictions2 = pd.read_feather(filen)

print("\ntrain set percentages\n____________________")
print(
    100 * train_predictions.cellassign_types.value_counts() / train_predictions.shape[0]
)


print("full set\n____________________")

print(
    100 * full_predictions.cellassign_types.value_counts() / full_predictions.shape[0]
)
# %%


train_predictions.index = train_predictions["cells"]

full_predictions.index = full_predictions["cells"]

# %%

train_cells = set(train_predictions["cells"])
full_predictions['train'] = full_predictions['cells'].apply(lambda x: x in train_cells)

# merge train and full predictions on cells
# name the add "train"  to the columns from train_predictions"

merged_predictions = pd.merge(full_predictions.reset_index(drop=True), train_predictions.reset_index(drop=True), on="cells", how="right", suffixes=("", "_train"))

merged_predictions = pd.merge(full_predictions, train_predictions,  how="right", suffixes=("", "_train"))


# %%

idx = train_predictions.cells

tmp = full_predictions[ train_predictions.cells == idx]


# %%
from sklearn.metrics import precision_score, recall_score, f1_score

import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion(
    df: pd.DataFrame,
    pred_key: str = "cellassign_types",
    cell_type_key: str = "cellassign_types_train",
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

    # HACK:  this is nasty... but it should work.
    # Keep the first 'pred' and all other columns
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # TODO: fix the problem upstream...

    # Calculate precision, recall, and F1-score
    prec = precision_score(df[cell_type_key], df[pred_key], average="macro")
    rec = recall_score(df[cell_type_key], df[pred_key], average="macro")
    f1 = f1_score(df[cell_type_key], df[pred_key], average="macro")
    acc = (df[pred_key] == df[cell_type_key]).mean()

    confusion_matrix = pd.crosstab(
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
    plt.tight_layout()


# %%

plot_confusion(merged_predictions)
# %%
2000 words two figures
brief report
# %%
