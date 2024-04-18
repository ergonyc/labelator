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

# ns_top_genes = [20_000, 10_000, 5_000, 3_000, 2_000, 1_000]
# ds_names = ["20k", "10k", "5k", "3k", "2k", "1k"]

ns_top_genes = [10_000, 5_000, 3_000, 2_000, 1_000]
ds_names = ["10k", "5k", "3k", "2k", "1k"]

# ns_top_genes = [5_000, 3_000, 1_000]
# ds_names = ["5k", "3k", "1k"]

# ns_top_genes = [20_000, 10_000]
# ds_names = ["20k", "10k"]


# In[ ]:

all_genes = {}
train_filen = raw_data_path / XYLENA2_TRAIN
# In[ ]:
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):

    genes = {}
    print(f"{ds_name}, seurat_v3")
    train_ad = ad.read_h5ad(train_filen)

    # process the "train" & "test" AnnData objects
    sc.pp.highly_variable_genes(
        train_ad,
        batch_key="sample",
        flavor="seurat_v3",
        n_top_genes=n_top_gene,
        subset=True,
        inplace=True,
    )
    genes["seurat_v3"] = train_ad.var_names.to_list()

    train_ad = ad.read_h5ad(train_filen)
    # train_ad = ad.read_h5ad(train_filen)
    print(f"{ds_name}, scvi")
    scvi.data.poisson_gene_selection(
        train_ad,
        n_top_genes=n_top_gene,
        batch_key="sample",
        subset=True,
        inplace=True,
    )
    genes["scvi"] = train_ad.var_names.to_list()

    train_ad = ad.read_h5ad(train_filen)
    print(f"{ds_name}, seurat")
    sc.pp.normalize_total(train_ad, target_sum=1e4)
    sc.pp.log1p(train_ad)
    sc.pp.highly_variable_genes(
        train_ad,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.25,
        n_top_genes=n_top_gene,
        batch_key="sample",
        subset=True,
        inplace=True,
    )
    genes["seurat"] = train_ad.var_names.to_list()

    train_ad = ad.read_h5ad(train_filen)

    print(f"{ds_name}, pearson_residuals")
    sc.experimental.pp.highly_variable_genes(
        train_ad,
        n_top_genes=n_top_gene,
        batch_key="sample",
        flavor="pearson_residuals",
        check_values=True,
        layer=None,
        subset=True,
        inplace=True,
    )
    genes["pearson_residuals"] = train_ad.var_names.to_list()

    all_genes[ds_name] = genes


with open("all_genes.pkl", "wb") as f:
    pickle.dump(all_genes, f)


# In[ ]:
grp_all_genes = {}

ns_top_genes = [10_000, 5_000, 3_000, 1_000]
ds_names = ["10k", "5k", "3k", "1k"]


for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    print(ds_name)
    genes = {}
    filen = raw_data_path / XYLENA2_TRAIN
    adata = ad.read_h5ad(filen)
    print(adata)
    scvi.data.poisson_gene_selection(
        adata,
        n_top_genes=n_top_gene,
        batch_key="sample",
        subset=True,
        inplace=True,
    )
    genes["train"] = adata.var_names.to_list()

    filen = raw_data_path / XYLENA2_TEST
    adata = ad.read_h5ad(filen)
    print(adata)
    scvi.data.poisson_gene_selection(
        adata,
        n_top_genes=n_top_gene,
        batch_key="sample",
        subset=True,
        inplace=True,
    )
    genes["test"] = adata.var_names.to_list()

    filen = raw_data_path / XYLENA2_QUERY
    adata = ad.read_h5ad(filen)
    print(adata)
    scvi.data.poisson_gene_selection(
        adata,
        n_top_genes=n_top_gene,
        batch_key="sample",
        subset=True,
        inplace=True,
    )
    genes["query"] = adata.var_names.to_list()

    grp_all_genes[ds_name] = genes


with open("grp_genes.pkl", "wb") as f:
    pickle.dump(grp_all_genes, f)


# # In[ ]:
# In[ ]:
# # use default scanpy for the pca
# sc.pp.pca(train_ad)
# train_filen = data_path / XYLENA2_TRAIN
# train_ad.write_h5ad(train_filen)

filen = raw_data_path / XYLENA2_TRAIN
adata = ad.read_h5ad(filen)

# In[ ]:
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

# In[ ]:
filen = raw_data_path / XYLENA2_TEST
adata = ad.read_h5ad(filen)
hvgs_test = sc.experimental.pp.highly_variable_genes(
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
filen = raw_data_path / XYLENA2_QUERY
adata = ad.read_h5ad(filen)
hvgs_query = sc.experimental.pp.highly_variable_genes(
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


hvgs_train.sort_values(
    ["highly_variable_nbatches", "highly_variable_rank"],
    ascending=[False, True],
    na_position="last",
    inplace=True,
)

hvgs_test.sort_values(
    ["highly_variable_nbatches", "highly_variable_rank"],
    ascending=[False, True],
    na_position="last",
    inplace=True,
)

hvgs_query.sort_values(
    ["highly_variable_nbatches", "highly_variable_rank"],
    ascending=[False, True],
    na_position="last",
    inplace=True,
)


# In[ ]:
for ds_name, n_top_gene in zip(ds_names, ns_top_genes):
    genes = {}
    genes["train"] = hvgs_train.iloc[:n_top_gene].index.to_list()
    genes["test"] = hvgs_test.iloc[:n_top_gene].index.to_list()
    genes["query"] = hvgs_query.iloc[:n_top_gene].index.to_list()
    grp_all_genes[ds_name] = genes


with open("grp_genes_pr.pkl", "wb") as f:
    pickle.dump(grp_all_genes, f)

# In[ ]:

# filen = raw_data_path / XYLENA2_TEST
# adata = ad.read_h5ad(filen)
# print(adata)
# sc.experimental.pp.highly_variable_genes(
#     train_ad,
#     n_top_genes=n_top_gene,
#     batch_key="sample",
#     flavor="pearson_residuals",
#     check_values=True,
#     layer=None,
#     subset=True,
#     inplace=True,
# )

# genes["test"] = adata.var_names.to_list()

# filen = raw_data_path / XYLENA2_QUERY
# adata = ad.read_h5ad(filen)
# print(adata)
# sc.experimental.pp.highly_variable_genes(
#     train_ad,
#     n_top_genes=n_top_gene,
#     batch_key="sample",
#     flavor="pearson_residuals",
#     check_values=True,
#     layer=None,
#     subset=True,
#     inplace=True,
# )

# genes["query"] = adata.var_names.to_list()


# sc.experimental.pp.highly_variable_genes(
#     train_ad,
#     n_top_genes=n_top_gene,
#     batch_key="sample",
#     flavor="pearson_residuals",
#     check_values=True,
#     layer=None,
#     subset=True,
#     inplace=True,
# )


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

grp_all_genes = pickle.load(open("grp_genes_pr.pkl", "rb"))

all_genes = pickle.load(open("all_genes.pkl", "rb"))


# %%


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


ns_top_genes = [10_000, 5_000, 3_000, 1_000]
ds_names = ["10k", "5k", "3k", "1k"]


for ds_name, n_top_genes in zip(ds_names, ns_top_genes):
    print("grp_all_genes")
    print(ds_name)
    compare_genes(grp_all_genes[ds_name], labels=["train", "query"])
    compare_genes(grp_all_genes[ds_name], labels=["test", "query"])
    compare_genes(grp_all_genes[ds_name], labels=["train", "test"])


ns_top_genes = [20_000, 10_000, 5_000, 3_000, 1_000]
ds_names = ["20k", "10k", "5k", "3k", "1k"]

for ds_name, n_top_genes in zip(ds_names, ns_top_genes):
    print("all_genes")
    print(ds_name)
    compare_genes(all_genes[ds_name], labels=["seurat", "pearson_residuals"])

    compare_genes(all_genes[ds_name], labels=["seurat", "seurat_v3"])
    compare_genes(all_genes[ds_name], labels=["seurat", "scvi"])
    compare_genes(all_genes[ds_name], labels=["scvi", "pearson_residuals"])
    compare_genes(all_genes[ds_name], labels=["scvi", "seurat_v3"])

    # genes["train"] = adata.var_names.to_list()

    # filen = ds_path / XYLENA2_TEST
    # adata = ad.read_h5ad(filen)
    # scvi.data.poisson_gene_selection(adata, n_top_genes=n_top_genes)
    # genes["test"] = adata.var_names.to_list()

    # filen = ds_path / XYLENA2_QUERY
    # adata = ad.read_h5ad(filen)
    # scvi.data.poisson_gene_selection(adata, n_top_genes=n_top_genes)
    # genes["query"] = adata.var_names.to_list()

    # grp_all_genes[ds_name] = genes


# %%
