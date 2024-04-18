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

sys.path.append(os.path.abspath("/media/ergonyc/Projects/SingleCell/labelator/"))

from lbl8r.model.utils._data import transfer_pcs

from lbl8r._constants import *

# In[ ]:
import numpy as np
import scvi

# ## create train and query datasets.

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

XYLENA2_FULL_LABELS = "full_labels.csv"
XYLENA2_FULL_CELLASSIGN = "full_cellassign.pkl"

## load raw data.
# In[ ]:
root_path = Path.cwd()
data_path = root_path / XYLENA2_PATH
raw_data_path = root_path / XYLENA2_RAW_PATH


# In[ ]: Load raw data
raw_filen = raw_data_path / XYLENA2_RAW_ANNDATA
raw_ad = ad.read_h5ad(raw_filen)

# In[ ]:
# # force sparse...
# raw_ad.X = sp.csr_matrix(raw_ad.X)


# In[ ]:
#  2. copy for cellassign
#  1. load marker_genes
filen = raw_data_path / "celltype_marker_table2.csv"

markers = pd.read_csv(filen, index_col=0)
# # In[ ]:

# # defensive
# markers = markers[~markers.index.duplicated(keep="first")].rename_axis(index=None)


# # bdata = adata[:, markers.index].copy() #
# bdata = raw_ad[:, raw_ad.var.index.isin(markers.index)].copy()

# #  3. get size_factor and noise

# lib_size = bdata.X.sum(1)  # type: ignore
# bdata.obs["size_factor"] = lib_size / np.mean(lib_size)
# # In[ ]:
# #  4. model = CellAssign(bdata, marker_genes)
# scvi.external.CellAssign.setup_anndata(
#     bdata,
#     size_factor_key="size_factor",
#     batch_key="sample",
#     layer=None,  #'counts',
#     # continuous_covariate_keys=noise
# )

# # In[ ]:
# #  5. model.train()
# model = scvi.external.CellAssign(bdata, markers)
# plan_args = {"lr_factor": 0.05, "lr_patience": 20, "reduce_lr_on_plateau": True}
# model.train(
#     max_epochs=1000,
#     accelerator="gpu",
#     early_stopping=True,
#     plan_kwargs=plan_args,
#     early_stopping_patience=40,
# )
# # In[ ]:
# #  6. model.predict()
# bdata.obs["cell_type"] = model.predict().idxmax(axis=1).values

# # In[ ]:
# # 7. transfer cell_type to adata
# raw_ad.obs["cell_type"] = bdata.obs["cell_type"]

# #  8. save model & artificts
# predictions = (
#     bdata.obs[["sample", "cell_type"]].reset_index().rename(columns={"index": "cells"})
# )
# predictions.to_csv(
#     raw_data_path / XYLENA2_GROUND_TRUTH, index=False
# )  # # pred_file = "cellassign_predictions.csv"

# In[ ]: collect metadata
ground_truth = pd.read_csv(raw_data_path / XYLENA2_FULL_LABELS)  # XYLENA2_GROUND_TRUTH)
obs = raw_ad.obs

# In[ ]: get the train/test splits
clean_samples_path = raw_data_path / "Model Combinations - clean_samples_138.csv"
clean_samples = pd.read_csv(clean_samples_path)
# all_samples_path = "/content/drive/MyDrive/SingleCellModel/Model Combinations - all_samples_199.csv"
# all_samples = pd.read_csv(all_samples_path)
dirty_samples_path = raw_data_path / "Model Combinations - dirty_samples_61.csv"
dirty_samples = pd.read_csv(dirty_samples_path)
test_samples_path = raw_data_path / "Model Combinations - testing_set_41.csv"
test_samples = pd.read_csv(test_samples_path)
train_samples_path = raw_data_path / "Model Combinations - training_set_98.csv"
train_samples = pd.read_csv(train_samples_path)

# In[ ]:
newmeta = obs.join(ground_truth.set_index("cells"), lsuffix="", rsuffix="_other")

newmeta["clean"] = [s in set(clean_samples["sample"]) for s in newmeta["sample"]]
newmeta["test"] = [s in set(test_samples["sample"]) for s in newmeta["sample"]]
newmeta["train"] = [s in set(train_samples["sample"]) for s in newmeta["sample"]]
newmeta["dirty"] = [s in set(dirty_samples["sample"]) for s in newmeta["sample"]]

# In[ ]:
newmeta["query"] = ~(newmeta["test"] | newmeta["train"])


# In[ ]:
newmeta["_cell_type"] = newmeta["cell_type"]
newmeta["cell_type"] = newmeta["cellassign_types"]

# In[ ]:

# newmeta["cell_type"] = newmeta["cellassign_types"]
# In[ ]:
# update anndata - only keep the metadata we need/want
raw_ad.obs = newmeta[
    [
        "total_counts",
        "total_counts_rb",
        "pct_counts_rb",
        "total_counts_mt",
        "pct_counts_mt",
        "doublet_score",
        "batch",
        "cohort",
        "sample",
        "n_genes_by_counts",
        "counts_deviation_score",
        "nCount_RNA",
        "nFeature_RNA",
        "S.Score",
        "G2M.Score",
        "Phase",
        "sample_other",
        "cell_type",
        "_cell_type",
        "train",
        "test",
        "query",
    ]
]

# In[ ]: save updated full AnnData object with updated metadata (obs)
outfilen = raw_data_path / XYLENA2_FULL

raw_ad.write_h5ad(outfilen)

# In[ ]:  make the train adata
train_ad = raw_ad[raw_ad.obs["train"]]

raw_train_filen = raw_data_path / XYLENA2_TRAIN
train_ad.write_h5ad(raw_train_filen)
del train_ad

# In[ ]:  make test anndata
test_ad = raw_ad[raw_ad.obs["test"]]

raw_test_filen = raw_data_path / XYLENA2_TEST
test_ad.write_h5ad(raw_test_filen)

del test_ad
# In[ ]:  read full anndata

full_filen = raw_data_path / XYLENA2_FULL
raw_ad = ad.read_h5ad(full_filen, backed="r")
# In[ ]:  make query anndata

query_ad = raw_ad[raw_ad.obs["query"]]
# In[ ]:
raw_query_filen = raw_data_path / XYLENA2_QUERY
query_ad.write_h5ad(raw_query_filen)
del query_ad

# In[ ]:

del raw_ad

# In[ ]:
#####################
# ns_top_genes = [20_000, 10_000, 5_000, 3_000, 2_000, 1_000]
# ds_names = ["20k", "10k", "5k", "3k", "2k", "1k"]

ns_top_genes = [10_000, 5_000, 3_000, 2_000, 1_000]
ds_names = ["10k", "5k", "3k", "2k", "1k"]

marker_genes = markers.index.to_list()
# In[ ]:
full_filen = raw_data_path / XYLENA2_FULL
adata = ad.read_h5ad(full_filen)

hvgs_full = sc.experimental.pp.highly_variable_genes(
    adata,
    n_top_genes=20_000,
    batch_key="sample",
    flavor="pearson_residuals",
    check_values=True,
    layer=None,
    subset=False,
    inplace=False,
)
# In[ ]:
##  loess fails for some batches... so we wil NOT use the desired default below
# # DEFAULT to seurat_v3 feature selection
# hvgs_full =  sc.pp.highly_variable_genes(
#         adata,
#         batch_key="sample",
#         flavor="seurat_v3",
#         n_top_genes=10_000,
#         subset=False,
#         inplace=False,
#     )
# # process the "train" & "test" AnnData objects

hvgs_full.to_csv(raw_data_path / XYLENA2_FULL_HVG)


# In[ ]:
train_filen = raw_data_path / XYLENA2_TRAIN
adata = ad.read_h5ad(train_filen)

hvgs_train = sc.experimental.pp.highly_variable_genes(
    adata,
    n_top_genes=20_000,
    batch_key="sample",
    flavor="pearson_residuals",
    check_values=True,
    layer=None,
    subset=False,
    inplace=False,
)
# process the "train" & "test" AnnData objects

hvgs_train.to_csv(raw_data_path / XYLENA2_TRAIN_HVG)

# In[ ]:

# raw_train_filen = raw_data_path / XYLENA2_TRAIN
# adata = ad.read_h5ad(raw_train_filen)

# hvgs_train = sc.experimental.pp.highly_variable_genes(
#     adata,
#     n_top_genes=20_000,
#     batch_key="sample",
#     flavor="pearson_residuals",
#     check_values=True,
#     layer=None,
#     subset=False,
#     inplace=False,
# )
# hvgs_train.to_csv(raw_data_path / XYLENA2_TRAIN_HVG)
# In[ ]:
# hvgs_train = pd.read_csv(raw_data_path / XYLENA2_TRAIN_HVG, index_col=0)
hvgs_full = pd.read_csv(raw_data_path / XYLENA2_FULL_HVG, index_col=0)
hvgs_train = pd.read_csv(raw_data_path / XYLENA2_TRAIN_HVG, index_col=0)


## TODO: update to new marker genes tables....

# In[ ]:
# curate the gene list with our marker genes at the top.
# markers = pd.read_csv("celltype_marker_table.csv", index_col=0)
# mset = set(markers.index)
# OLD
markers = pd.read_csv("celltype_marker_table.csv", index_col=0)
# NEW
markers = pd.read_csv(raw_data_path / "celltype_marker_table2.csv", index_col=0)

# # defensive
# markers = markers[~markers.index.duplicated(keep="first")].rename_axis(index=None)

# hvgs = set(hvgs_full.index)
# In[ ]:
# hvgs_full.loc[markers.index, 'highly_variable_rank'] = 1.
hvgs_full.loc[markers.index, "highly_variable_nbatches"] = 347.0

# In[ ]:
# Sort genes by how often they selected as hvg within each batch and
# break ties with median rank of residual variance across batches
hvgs_full.sort_values(
    ["highly_variable_nbatches", "highly_variable_rank"],
    ascending=[False, True],
    na_position="last",
    inplace=True,
)

# hvgs_train.sort_values(
#     ["highly_variable_nbatches", "highly_variable_rank"],
#     ascending=[False, True],
#     na_position="last",
#     inplace=True,
# )
# In
gene_list = hvgs_full.iloc[:20_000].copy()

gene_list["marker"] = False
gene_list.loc[markers.index, "marker"] = True

gene_list.to_csv(raw_data_path / "xyl2_full_hvg.csv")

# In[ ]:

gene_list = pd.read_csv(raw_data_path / "xyl2_full_hvg.csv", index_col=0)

gene_cuts = {}
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    gene_cuts[ds_name] = gene_list.iloc[:n_top_gene].index.to_list()


# In[ ]:

raw_train_filen = raw_data_path / XYLENA2_TRAIN
train_ad = ad.read_h5ad(raw_train_filen)


for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    keep_genes = gene_cuts[ds_name]
    print(f"Keeping {len(keep_genes)} genes")
    # load the raw test_ad
    train_ad = train_ad[:, keep_genes]

    ds_path = data_path.parent / f"{data_path.name}{ds_name}"
    train_filen = ds_path / XYLENA2_TRAIN
    train_ad.write_h5ad(train_filen)
    print(f"Saved {train_filen}")

del train_ad
# In[ ]:
raw_test_filen = raw_data_path / XYLENA2_TEST
test_ad = ad.read_h5ad(raw_test_filen)
# subset
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    keep_genes = gene_cuts[ds_name]
    print(f"Keeping {len(keep_genes)} genes")
    # load the raw test_ad
    test_ad = test_ad[:, keep_genes]

    # save the test_ad with the highly variable genes
    ds_path = data_path.parent / f"{data_path.name}{ds_name}"
    test_filen = ds_path / XYLENA2_TEST
    test_ad.write_h5ad(test_filen)
    print(f"Saved {test_filen}")

del test_ad
# In[ ]:
# load the raw quest_ad
raw_query_filen = raw_data_path / XYLENA2_QUERY
query_ad = ad.read_h5ad(raw_query_filen)
# subset
# In[ ]:
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    keep_genes = gene_cuts[ds_name]
    print(f"Keeping {len(keep_genes)} genes")
    query_ad = query_ad[:, keep_genes]

    # save the query_ad with the highly variable genes
    ds_path = data_path.parent / f"{data_path.name}{ds_name}"
    query_filen = ds_path / XYLENA2_QUERY
    query_ad.write_h5ad(query_filen)
    print(f"Saved {query_filen}")
del query_ad

# In[ ]:
# # In[ ]:
# # use default scanpy for the pca
# sc.pp.pca(train_ad)
# train_filen = data_path / XYLENA2_TRAIN
# train_ad.write_h5ad(train_filen)

# # In[ ]:
# # load the raw test_ad
# raw_test_filen = raw_data_path / XYLENA2_TEST
# test_ad = ad.read_h5ad(raw_test_filen)

# # In[ ]: # now we need to copy the PCs to the test set and compute loadings.
# test_ad = transfer_pcs(test_ad, train_ad)
# # In[ ]:
# test_ad.write_h5ad(test_filen)

# # In[ ]:
# del test_ad
# del train_ad

# # Optional save memory/disk space by using sparse matrices.  Models will train / infer slower
# # In[ ]:
# train_ad = ad.read_h5ad(train_filen)


# In[ ]:
def compare_genes(genes, labels=("genes1", "genes2")):
    g1 = set(genes[labels[0]])
    g2 = set(genes[labels[1]])
    print(f"number of genes in {labels[0]}: {len(g1)}")
    print(f"number of genes in {labels[1]}: {len(g2)}")

    print(f"{labels[0]} - {labels[1]}: {len(g1 - g2)}")
    print(f"{labels[1]} - {labels[0]} {len(g2 - g1)}")
    print(f"{labels[0]} & {labels[1]}: {len(g1 & g2)}")
    print(f"{labels[0]} | {labels[1]}: {len(g1 | g2)}")


# %%


for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    print("gene_cuts")
    print(ds_name)
    compare_genes(gene_cuts[ds_name], labels=["train", "full"])

# %%


for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    tset = set(gene_cuts[ds_name]["train"])
    fset = set(gene_cuts[ds_name]["full"])
    print(f"mset-tset: {len(mset - tset)}")
    print(f"mset-fset: {len(mset - fset)}")


# %%
