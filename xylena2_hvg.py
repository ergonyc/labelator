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
raw_filen = raw_data_path / XYLENA2_RAW_ANNDATA
raw_ad = ad.read_h5ad(raw_filen)


# In[ ]:
#####################

ns_top_genes = [20_000, 10_000, 5_000, 3_000, 1_000]
ds_names = ["20k", "10k", "5k", "3k", "1k"]

ns_top_genes = [5_000, 3_000, 1_000]
ds_names = ["5k", "3k", "1k"]

# In[ ]:

all_genes = {}
# In[ ]:
for ds_name, n_top_genes in zip(ds_names, ns_top_genes):

    genes = {}
    ds_path = root_path / f"{XYLENA2_PATH}{ds_name}"

    train_filen = ds_path / XYLENA2_TRAIN
    train_ad = ad.read_h5ad(train_filen)

    # process the "train" & "test" AnnData objects
    sc.pp.highly_variable_genes(
        train_ad,
        batch_key="sample",
        subset=True,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )
    genes["seurat_v3"] = train_ad.var_names.to_list()

    train_ad = ad.read_h5ad(train_filen)
    scvi.data.poisson_gene_selection(train_ad, n_top_genes=n_top_genes)
    genes["scvi"] = train_ad.var_names.to_list()

    train_ad = ad.read_h5ad(train_filen)
    sc.pp.normalize_total(train_ad, target_sum=1e4)
    sc.pp.log1p(train_ad)
    sc.pp.highly_variable_genes(
        train_ad, min_mean=0.0125, max_mean=3, min_disp=0.25, n_top_genes=n_top_genes
    )
    genes["seurat"] = train_ad.var_names.to_list()

    train_ad = ad.read_h5ad(train_filen)
    sc.experimental.pp.highly_variable_genes(
        train_ad,
        theta=100,
        clip=None,
        n_top_genes=n_top_genes,
        batch_key=None,
        chunksize=1000,
        flavor="pearson_residuals",
        check_values=True,
        layer=None,
        subset=True,
        inplace=True,
    )
    genes["pearson_residuals"] = train_ad.var_names.to_list()

    all_genes["ds_name"] = genes

import pickle

with open("all_genes.pkl", "wb") as f:
    pickle.dump(all_genes, f)


# In[ ]:
grp_all_genes = {}

for ds_name, n_top_genes in zip(ds_names, ns_top_genes):
    ds_path = root_path / f"{XYLENA2_PATH}{ds_name}"

    genes = {}
    filen = ds_path / XYLENA2_TRAIN
    adata = ad.read_h5ad(filen)
    scvi.data.poisson_gene_selection(adata, n_top_genes=n_top_genes)
    genes["train"] = adata.var_names.to_list()

    filen = ds_path / XYLENA2_TEST
    adata = ad.read_h5ad(filen)
    scvi.data.poisson_gene_selection(adata, n_top_genes=n_top_genes)
    genes["test"] = adata.var_names.to_list()

    filen = ds_path / XYLENA2_QUERY
    adata = ad.read_h5ad(filen)
    scvi.data.poisson_gene_selection(adata, n_top_genes=n_top_genes)
    genes["query"] = adata.var_names.to_list()

    grp_all_genes["ds_name"] = genes


with open("grp_genes.pkl", "wb") as f:
    pickle.dump(grp_all_genes, f)


# # In[ ]:
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


# # In[ ]:

# %%
