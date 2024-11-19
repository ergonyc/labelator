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


sys.path.append(Path.cwd().parent.as_posix())

# gs://sc-labelator-data/full_anndata_object.h5ad
# from google.cloud import storage

# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     """Uploads a file to the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(source_file_name)
#     print(f"File {source_file_name} uploaded to {destination_blob_name}.")

# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """Downloads a blob from the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#     print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

# # upload_blob(bucket_name, upload_file, blob_name)
# # download_blob(bucket_name, blob_name, download_file)

# def load_ad_bucket(bucket_name, blob_name):
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     return ad.read_h5ad(blob.download_as_bytes())

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
XYLENA2_GROUNDTRUTH_LABELS = "cluster_labels.csv"

XYLENA2_FULL_CELLASSIGN = "full_cellassign_model"

## load raw data.

# In[ ]:
root_path = Path.cwd().parent
data_path = root_path / XYLENA2_PATH
raw_data_path = root_path / XYLENA2_RAW_PATH
# In[ ]: Load raw data
########################
# 0. LOAD RAW DATAgit a
########################
raw_filen = raw_data_path / XYLENA2_RAW_ANNDATA
# raw_ad = ad.read_h5ad(raw_filen)
raw_ad = ad.read_h5ad("/data/scdata/xylena_raw/full_anndata_object.h5ad", backed="r")

# # Replace with your bucket name and file paths
# bucket_name = "gs://sc-labelator-data/"
# download_file = "full_anndata_object.h5ad"
# raw_ad = load_ad_bucket(bucket_name, download_file)

# In[ ]:
# it looks like the unlabeled samples (ie. no ground truth labels) are for total count < 500

# In[ ]:
# load ground truth labels
filen = raw_data_path / XYLENA2_GROUNDTRUTH_LABELS
ground_truth = pd.read_csv(filen)
ground_truth.set_index("barcodes", inplace=True)
# In[ ]: collect metadata
# raw_ad.var_names_make_unique()
obs = raw_ad.obs

# In[ ]: get the train/test splits
newmeta = obs.join(ground_truth, lsuffix="", rsuffix="_other")

# In[ ]:
newmeta["unlabeled"] = newmeta["celltype"].isna()
newmeta["labeled"] = newmeta["celltype"].notna()

# In[ ]:
newmeta["labeled_sample"] = False
newmeta["unlabeled_sample"] = False
newmeta["partlabeled_sample"] = False
# In[ ]:
usamples = newmeta["sample"].unique()
# df = pd.DataFrame(index=usamples)

for samp in usamples:
    subset = newmeta[newmeta["sample"] == samp]

    newmeta.loc[subset.index, "labeled_sample"] = subset["labeled"].all()
    newmeta.loc[subset.index, "unlabeled_sample"] = subset["unlabeled"].all()
    newmeta.loc[subset.index, "partlabeled_sample"] = (
        subset["unlabeled"].any() & subset["labeled"].any()
    )
    newmeta.loc[subset.index, "frac_labeled"] = (
        subset["labeled"].sum() / subset.shape[0]
    )
    newmeta.loc[subset.index, "n_cells"] = subset.shape[0]


# In[ ]:
# arbitrarily choose the samples with > 55% labeled cells as the training set
newmeta["tt"] = newmeta["frac_labeled"] > 0.55

newmeta["query"] = ~newmeta["tt"]

# In[ ]:
# do train test split...
tt_samples = newmeta.loc[newmeta["tt"], "sample"].unique().tolist()
n_tt_samp = int(len(tt_samples) * 0.7)

train_samples = tt_samples[:n_tt_samp]
test_samples = tt_samples[n_tt_samp:]
# In[ ]:
newmeta["train"] = newmeta["sample"].isin(train_samples)
newmeta["test"] = newmeta["sample"].isin(test_samples)

# In[ ]:
newmeta["cell_type"] = newmeta["celltype"]

# In[ ]:
newmeta["cell_type"].fillna("UNKNOWN", inplace=True)
# newmeta["cell_type"] = newmeta["cellassign_types"]

# In[ ]:
newmeta["n_counts"] = newmeta["nCount_RNA"]
newmeta["n_genes"] = newmeta["nFeature_RNA"]
newmeta["s_score"] = newmeta["S.Score"]
newmeta["g2m_score"] = newmeta["G2M.Score"]
newmeta["phase"] = newmeta["Phase"]


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
        "n_cells",
        "batch",
        "cohort",
        "sample",
        "n_genes_by_counts",
        "counts_deviation_score",
        "n_counts",
        "n_genes",
        "s_score",
        "g2m_score",
        "phase",
        # "nCount_RNA",
        # "nFeature_RNA",
        # "S.Score",
        # "G2M.Score",
        # "Phase",
        "sample_other",
        "cell_type",
        "celltype",
        "train",
        "test",
        "query",
    ]
].copy()

# In[ ]: save updated full AnnData object with updated metadata (obs)
outfilen = data_path / XYLENA2_FULL

raw_ad.write_h5ad(outfilen)

# In[ ]:  read full anndata

full_filen = data_path / XYLENA2_FULL
raw_ad = ad.read_h5ad(full_filen, backed="r")

# In[ ]:  make the train adata
train_ad = raw_ad[raw_ad.obs["train"]]
# In[ ]:
raw_train_filen = data_path / XYLENA2_TRAIN
# train_ad = raw_ad[raw_ad.obs["train"]].copy(raw_train_filen)

train_ad.write_h5ad(raw_train_filen)
del train_ad

# In[ ]:  read full anndata
full_filen = data_path / XYLENA2_FULL
raw_ad = ad.read_h5ad(full_filen, backed="r")
# In[ ]:  make test anndata
test_ad = raw_ad[raw_ad.obs["test"]]

raw_test_filen = data_path / XYLENA2_TEST
test_ad.write_h5ad(raw_test_filen)

del test_ad
# In[ ]:  read full anndata

full_filen = data_path / XYLENA2_FULL
raw_ad = ad.read_h5ad(full_filen, backed="r")
# In[ ]:  make query anndata

query_ad = raw_ad[raw_ad.obs["query"]]
# In[ ]:
raw_query_filen = data_path / XYLENA2_QUERY
query_ad.write_h5ad(raw_query_filen)
del query_ad

# In[ ]:

del raw_ad

# In[ ]:
########################
# 2. HIGHLY VARIABLE GENES
########################
# ns_top_genes = [20_000, 10_000, 5_000, 3_000, 2_000, 1_000]
# ds_names = ["20k", "10k", "5k", "3k", "2k", "1k"]

ns_top_genes = [10_000, 5_000, 3_000, 2_000, 1_000]
ds_names = ["10k", "5k", "3k", "2k", "1k"]

# In[ ]:
full_filen = data_path / XYLENA2_FULL
# adata = ad.read_h5ad(full_filen,backed='r+')
adata = ad.read_h5ad(full_filen)

# In[ ]:
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

hvgs_full.to_csv(data_path / XYLENA2_FULL_HVG)


# In[ ]:
# hvgs_train = pd.read_csv(raw_data_path / XYLENA2_TRAIN_HVG, index_col=0)
hvgs_full = pd.read_csv(data_path / XYLENA2_FULL_HVG, index_col=0)


# In[ ]:

gene_list = pd.read_csv(data_path / XYLENA2_FULL_HVG, index_col=0)
ns_top_genes = [10_000, 5_000, 3_000, 2_000, 1_000]
ds_names = ["10k", "5k", "3k", "2k", "1k"]


gene_cuts = {}
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    gene_cuts[ds_name] = gene_list.iloc[:n_top_gene].index.to_list()


# In[ ]:
del adata


# In[ ]:


raw_train_filen = data_path / XYLENA2_TRAIN
train_ad = ad.read_h5ad(raw_train_filen)


for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    keep_genes = gene_cuts[ds_name]
    print(f"Keeping {len(keep_genes)} genes")
    # load the raw test_ad
    train_ad = train_ad[:, keep_genes]

    # ds_path = data_path.parent / f"{data_path.name}{ds_name}"
    ds_path = data_path / ds_name
    ds_path.mkdir(exist_ok=True)

    # compute_pcs(train_ad, save_path=ds_path, set_name=raw_train_filen.stem)

    train_filen = ds_path / XYLENA2_TRAIN
    train_ad.write_h5ad(train_filen)
    print(f"Saved {train_filen}")

# del train_ad
# In[ ]:
raw_test_filen = data_path / XYLENA2_TEST
test_ad = ad.read_h5ad(raw_test_filen)
# subset
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    keep_genes = gene_cuts[ds_name]
    print(f"Keeping {len(keep_genes)} genes")
    # load the raw test_ad
    test_ad = test_ad[:, keep_genes]

    # save the test_ad with the highly variable genes
    # ds_path = data_path.parent / f"{data_path.name}{ds_name}"
    ds_path = data_path / ds_name

    # # get loadings
    # pcs = load_pcs(ds_path)
    # X_pca = transfer_pca(test_ad, pcs)
    # pcs_name = f"X_pca_{raw_test_filen.stem}.npy"
    # dump_x_pca(X_pca, ds_path, pcs_name=pcs_name)

    test_filen = ds_path / XYLENA2_TEST
    test_ad.write_h5ad(test_filen)
    print(f"Saved {test_filen}")

del test_ad
# In[ ]:
# load the raw quest_ad
raw_query_filen = data_path / XYLENA2_QUERY
query_ad = ad.read_h5ad(raw_query_filen)
# subset
# In[ ]:
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    keep_genes = gene_cuts[ds_name]
    print(f"Keeping {len(keep_genes)} genes")
    query_ad = query_ad[:, keep_genes]

    # save the query_ad with the highly variable genes
    # ds_path = data_path.parent / f"{data_path.name}{ds_name}"
    ds_path = data_path / ds_name

    # # get loadings
    # pcs = load_pcs(ds_path)
    # X_pca = transfer_pca(query_ad, pcs)

    # pcs_name = f"X_pca_{raw_query_filen.stem}.npy"
    # dump_x_pca(X_pca, ds_path, pcs_name=pcs_name)

    query_filen = ds_path / XYLENA2_QUERY
    query_ad.write_h5ad(query_filen)
    print(f"Saved {query_filen}")
del query_ad
<<<<<<< Updated upstream

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

mset = set(gene_list.index)
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    tset = set(gene_cuts[ds_name]["train"])
    fset = set(gene_cuts[ds_name]["full"])
    print(f"mset-tset: {len(mset - tset)}")
    print(f"mset-fset: {len(mset - fset)}")


=======
>>>>>>> Stashed changes
