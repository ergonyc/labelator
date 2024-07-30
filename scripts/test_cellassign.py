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


sys.path.append(os.path.abspath("/home/ergonyc/Projects/SingleCell/labelator/"))

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


## load raw data.

# In[ ]:
root_path = Path.cwd().parent
data_path = root_path / XYLENA2_PATH
raw_data_path = root_path / XYLENA2_RAW_PATH

# In[ ]:
markers_path = root_path / "taxonomy/cellassign_markers.csv"
markers = pd.read_csv(markers_path, index_col=0)

# In[ ]:


def get_cell_types(adata, markers, batch_key=None, noise=None, seed=9627):

    #  2. copy for cellassign
    # bdata = adata[:, markers.index].copy() #
    bdata = adata[:, adata.var.index.isin(markers.index)].copy()

    #  3. get size_factor and noise
    lib_size = adata.X.sum(1)  # type: ignore
    bdata.obs["size_factor"] = lib_size / np.mean(lib_size)

    print("size_factor", bdata.obs["size_factor"].mean())
    #  4. model = CellAssign(bdata, marker_genes)

    scvi.external.CellAssign.setup_anndata(
        bdata,
        size_factor_key="size_factor",
        # batch_key="sample",
        batch_key=batch_key,
        layer=None,  #'counts',
        continuous_covariate_keys=noise,
    )

    #  5. model.train()
    scvi.settings.seed = seed
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

    preds["cellassign_types"] = preds.idxmax(axis=1).values

    # # 7. transfer cell_type to adata
    # adata.obs["cellassign_types"] = preds["cellassign_types"]

    if "cell_type" not in bdata.obs_keys():
        preds["cell_type"] = "NONE"
    else:
        preds["cell_type"] = bdata.obs["cell_type"].values

    preds["sample"] = bdata.obs["sample"].values
    preds["cell"] = bdata.obs.index.values
    preds.index = preds["cell"].values

    return preds, model


def get_cell_types_(adata, markers, model_path, batch_key=None, noise=None):

    #  2. copy for cellassign
    # bdata = adata[:, markers.index].copy() #
    bdata = adata[:, adata.var.index.isin(markers.index)].copy()

    #  3. get size_factor and noise
    lib_size = adata.X.sum(1)  # type: ignore
    bdata.obs["size_factor"] = lib_size / np.mean(lib_size)

    print("size_factor", bdata.obs["size_factor"].mean())
    #  4. model = CellAssign(bdata, marker_genes)

    scvi.external.CellAssign.setup_anndata(
        bdata,
        size_factor_key="size_factor",
        # batch_key="sample",
        batch_key=batch_key,
        layer=None,  #'counts',
        continuous_covariate_keys=noise,
    )

    #  5. model.train()
    model = scvi.external.CellAssign(bdata, markers)
    model.load(model_path, bdata)

    #  6. model.predict()
    preds = model.predict()

    preds["cellassign_types"] = preds.idxmax(axis=1).values

    # # 7. transfer cell_type to adata
    # adata.obs["cellassign_types"] = preds["cellassign_types"]

    if "cell_type" not in bdata.obs_keys():
        preds["cell_type"] = "NONE"
    else:
        preds["cell_type"] = bdata.obs["cell_type"]

    preds["sample"] = bdata.obs["sample"]
    preds["cell"] = bdata.obs.index.values
    preds.index = preds["cell"]

    return preds, model


# In[ ]: Load raw data


########################
# 0. LOAD TRAIN DATA
########################
filen = data_path / XYLENA2_FULL
adata = ad.read_h5ad(filen)

# In[ ]:
train_test_samp = adata.obs["train"] | adata.obs["test"]
noise = ["doublet_score", "pct_counts_mt", "pct_counts_rb"]  # aka "noise"

for samp_set in ["full"]:  # ["full", "clean"]:
    bdata = adata[train_test_samp] if samp_set == "clean" else adata

    for rep in [5, 6, 7]:
        predictions, model = get_cell_types(
            bdata, markers, batch_key="sample", noise=noise
        )

        filen = root_path / "testing" / f"{samp_set}{rep}_noise_predictions.feather"
        predictions.reset_index(drop=True).to_feather(filen)

        modelname = root_path / f"{samp_set}{rep}_noise_cellassign"
        model.save(modelname, overwrite=True)

        predictions, model = get_cell_types(
            bdata, markers, batch_key="sample", noise=None
        )

        filen = root_path / "testing" / f"{samp_set}{rep}_predictions.feather"
        predictions.reset_index(drop=True).to_feather(filen)

        modelname = root_path / f"{samp_set}{rep}_cellassign"
        model.save(modelname, overwrite=True)


# # In[ ]:
# train_test_samp = adata.obs["train"] | adata.obs["test"]
# noise = ['doublet_score', 'pct_counts_mt', 'pct_counts_rb'] # aka "noise"

# for samp_set in ["full", "clean"]:
#     bdata = adata[train_test_samp] if samp_set == "clean" else adata

#     for rep in range(2):

#         modelname = f"{samp_set}{rep}_noise_cellassign"
#         predictions, model = get_cell_types_(bdata, markers,modelname, batch_key="sample", noise=noise)

#         filen = f"{samp_set}{rep}_noise_predictions.feather"
#         predictions.reset_index(drop=True).to_feather(filen)

#         modelname = f"{samp_set}{rep}_cellassign"
#         predictions, model = get_cell_types(bdata, markers,modelname, batch_key="sample", noise=None)

#         filen = f"{samp_set}{rep}_predictions.feather"
#         predictions.reset_index(drop=True).to_feather(filen)


# # In[ ]:

# samp_set = "clean"
# rep = 0
# modelname = f"{samp_set}{rep}_noise_cellassign"
# bdata = adata[train_test_samp] if samp_set == "clean" else adata

# predictions, model = get_cell_types_(bdata, markers,modelname, batch_key="sample", noise=noise)


# In[ ]: summarize


samp_set = "full"
# samp_set = "clean"
noise = "_noise"
# noise = ""
rep = 5

A = f"{samp_set}{rep}{noise}"
filenA = root_path / f"testing/{A}_predictions.feather"
predictionsA = pd.read_feather(filenA)

rep = 6
B = f"{samp_set}{rep}{noise}"
filenB = root_path / f"testing/{B}_predictions.feather"
predictionsB = pd.read_feather(filenB)


print(f"\n{A} percentages\n____________________")
print(100 * predictionsA.cellassign_types.value_counts() / predictionsA.shape[0])

print(f"\n{B} percentages\n____________________")
print(100 * predictionsB.cellassign_types.value_counts() / predictionsB.shape[0])


rep = 7
C = f"{samp_set}{rep}{noise}"
filenC = root_path / f"testing/{C}_predictions.feather"
predictionsC = pd.read_feather(filenC)
print(f"\n{C} percentages\n____________________")
print(100 * predictionsC.cellassign_types.value_counts() / predictionsC.shape[0])


print(f"\n{A} counts\n____________________")
print(predictionsA.cellassign_types.value_counts())
print(f"\n{B} counts\n____________________")
print(predictionsB.cellassign_types.value_counts())
print(f"\n{C} counts\n____________________")
print(predictionsC.cellassign_types.value_counts())


# In[ ]:

merged_predictions = pd.merge(
    predictionsA, predictionsB, on="cell", how="right", suffixes=("_A", "_B")
)


# %%
from sklearn.metrics import precision_score, recall_score, f1_score

import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion(
    df: pd.DataFrame,
    title_caption: str = "Confusion matrix",
    pred_key: str = "cellassign_types_A",
    cell_type_key: str = "cellassign_types_B",
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
    plt.suptitle(title_caption)

    plt.tight_layout()


# %%


def print_count_subset(
    predictionsA: pd.DataFrame, predictionsB: pd.DataFrame, labs: list = ["A", "B"]
):

    merged_predictions = pd.merge(
        predictionsA, predictionsB, on="cell", how="left", suffixes=("_A", "_B")
    )

    summary = pd.DataFrame()
    key = f"{labs[0]}pct"
    summary[key] = (
        100 * predictionsA.cellassign_types.value_counts() / predictionsA.shape[0]
    )
    key = f"{labs[1]}pct"
    summary[key] = (
        100 * predictionsB.cellassign_types.value_counts() / predictionsB.shape[0]
    )

    # summary["Apct_"] = (
    #     100
    #     * merged_predictions.cellassign_types_A.value_counts()
    #     / merged_predictions.shape[0]
    # )
    key = f"{labs[1]}pct_"
    summary[key] = (
        100
        * merged_predictions.cellassign_types_B.value_counts()
        / merged_predictions.shape[0]
    )
    key = f"{labs[0]}"
    summary[key] = predictionsA.cellassign_types.value_counts()

    key = f"{labs[1]}_"
    summary[key] = merged_predictions.cellassign_types_B.value_counts()

    print(
        f" N samples {labs[0]}= {predictionsA.shape[0]}, N samples {labs[1]}= {predictionsB.shape[0]}"
    )
    print(summary)

    return summary


def print_count_group(
    predictionsA: pd.DataFrame, predictionsB: pd.DataFrame, labs: list = ["A", "B"]
):

    summary = pd.DataFrame()
    key = f"{labs[0]}pct"
    summary[key] = (
        100 * predictionsA.cellassign_types.value_counts() / predictionsA.shape[0]
    )
    key = f"{labs[1]}pct"
    summary[key] = (
        100 * predictionsB.cellassign_types.value_counts() / predictionsB.shape[0]
    )

    key = f"{labs[0]}"
    summary[key] = predictionsA.cellassign_types.value_counts()
    key = f"{labs[1]}"
    summary[key] = predictionsB.cellassign_types.value_counts()

    print(
        f" N samples {labs[0]}= {predictionsA.shape[0]}, N samples {labs[1]}= {predictionsB.shape[0]}"
    )
    print(summary)

    return summary


# %%
# 2000 words two figures
# brief report
# # %%

# samp_set = "full"
# samp_set = "clean"
# noise = "_noise"
# noise = ""
# rep = 0

# for samp_set in ["full", "clean"]:
#     for noise in ["_noise", ""]:
#         rep = 5
#         A = f"{samp_set}{rep}{noise}"
#         filenA = root_path / f"testing/{A}_predictions.feather"
#         predictionsA = pd.read_feather(filenA)

#         rep = 6
#         B = f"{samp_set}{rep}{noise}"
#         filenB = root_path / f"testing/{B}_predictions.feather"
#         predictionsB = pd.read_feather(filenB)

#         merged_predictions = pd.merge(
#             predictionsA, predictionsB, on="cell", how="right", suffixes=("_A", "_B")
#         )

#         summary = print_counts(predictionsA, predictionsB)
#         plot_confusion(merged_predictions, title_caption=f"{A} vs. {B}")

# %%

samp_set = "full"
samp_set = "clean"
noise = "_noise"
noise = ""
rep = 6

for samp_set in ["full", "clean"]:
    noise = "_noise"
    A = f"{samp_set}{rep}{noise}"
    filenA = root_path / f"testing/{A}_predictions.feather"
    predictionsA = pd.read_feather(filenA)

    noise = ""
    B = f"{samp_set}{rep}{noise}"
    filenB = root_path / f"testing/{B}_predictions.feather"
    predictionsB = pd.read_feather(filenB)

    merged_predictions = pd.merge(
        predictionsA, predictionsB, on="cell", how="right", suffixes=("_A", "_B")
    )
    print_count_group(predictionsA, predictionsB, labs=[A, B])
    plot_confusion(merged_predictions, title_caption=f"{A} vs. {B}")

# %%

samp_set = "full"
samp_set = "clean"
noise = "_noise"
noise = ""
rep = 6

for noise in ["_noise", ""]:
    samp_set = "clean"
    A = f"{samp_set}{rep}{noise}"
    filenA = root_path / f"testing/{A}_predictions.feather"
    predictionsA = pd.read_feather(filenA)

    samp_set = "full"
    B = f"{samp_set}{rep}{noise}"
    filenB = root_path / f"testing/{B}_predictions.feather"
    predictionsB = pd.read_feather(filenB)

    merged_predictions = pd.merge(
        predictionsA, predictionsB, on="cell", how="left", suffixes=("_A", "_B")
    )
    print_count_subset(predictionsA, predictionsB, labs=[A, B])
    plot_confusion(merged_predictions, title_caption=f"{A} vs. {B}")


# %%

samp_set = "full"
samp_set = "clean"
noise = "_noise"
noise = ""
rep = 1

for samp_set in ["full", "clean"]:
    noise = "_noise"
    A = f"{samp_set}{rep}{noise}"
    filenA = root_path / f"testing/{A}_predictions.feather"
    predictionsA = pd.read_feather(filenA)

    noise = ""
    B = f"{samp_set}{rep}{noise}"
    filenB = root_path / f"testing/{B}_predictions.feather"
    predictionsB = pd.read_feather(filenB)

    merged_predictions = pd.merge(
        predictionsA, predictionsB, on="cell", how="right", suffixes=("_A", "_B")
    )

    plot_confusion(merged_predictions, title_caption=f"{A} vs. {B}")

# %%

samp_set = "full"
samp_set = "clean"
noise = "_noise"
noise = ""
rep = 1

for noise in ["_noise", ""]:
    samp_set = "clean"
    A = f"{samp_set}{rep}{noise}"
    filenA = root_path / f"testing/{A}_predictions.feather"
    predictionsA = pd.read_feather(filenA)

    samp_set = "full"
    B = f"{samp_set}{rep}{noise}"
    filenB = root_path / f"testing/{B}_predictions.feather"
    predictionsB = pd.read_feather(filenB)

    merged_predictions = pd.merge(
        predictionsA, predictionsB, on="cell", how="left", suffixes=("_A", "_B")
    )

    plot_confusion(merged_predictions, title_caption=f"{A} vs. {B}")


# %%

samp_set = "clean"
noise = "_noise"
rep = 6

A = f"{samp_set}{rep}{noise}"
filenA = root_path / f"testing/{A}_predictions.feather"
predictionsA = pd.read_feather(filenA)


samp_set = "full"
rep = 6
B = f"{samp_set}{rep}{noise}"
filenB = root_path / f"testing/{B}_predictions.feather"
predictionsB = pd.read_feather(filenB)

merged_predictions = pd.merge(
    predictionsA, predictionsB, on="cell", how="left", suffixes=("_A", "_B")
)
plot_confusion(merged_predictions, title_caption=f"{A} vs. {B}")

# %%


print(f"\n{A} percentages\n____________________")
print(100 * predictionsA.cellassign_types.value_counts() / predictionsA.shape[0])

print(f"\n{B} percentages\n____________________")
print(100 * predictionsB.cellassign_types.value_counts() / predictionsB.shape[0])


print(f"\n{A} counts\n____________________")
print(predictionsA.cellassign_types.value_counts())
print(f"\n{B} counts\n____________________")
print(predictionsB.cellassign_types.value_counts())


# %%
merged_predictions = pd.merge(
    predictionsA, predictionsB, on="cell", how="left", suffixes=("_A", "_B")
)
plot_confusion(merged_predictions, title_caption=f"{A} vs. {B}")

# %%
samp_set = "clean"
noise = ""
rep = 6

A = f"{samp_set}{rep}{noise}"
filenA = root_path / f"testing/{A}_predictions.feather"
predictionsA = pd.read_feather(filenA)

rep = 1
B = f"{samp_set}{rep}{noise}"
filenB = root_path / f"testing/{B}_predictions.feather"
predictionsB = pd.read_feather(filenB)


merged_predictions = pd.merge(
    predictionsA, predictionsB, on="cell", how="left", suffixes=("_A", "_B")
)
plot_confusion(merged_predictions, title_caption=f"{A} vs. {B}")

# %%
