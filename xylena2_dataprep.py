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

## load raw data.

# In[ ]:
root_path = Path.cwd()
data_path = root_path / XYLENA2_PATH
raw_data_path = root_path / XYLENA2_RAW_PATH
# In[ ]: Load raw data
########################
# 0. LOAD RAW DATA
########################
raw_filen = raw_data_path / XYLENA2_RAW_ANNDATA
raw_ad = ad.read_h5ad(raw_filen)

# In[ ]:
########################
# 1. CELLTYPES FROM MARKER GENES
########################

#  2. copy for cellassign
#  1. load marker_genes
filen = raw_data_path / "celltype_marker_table2.csv"

markers = pd.read_csv(filen, index_col=0)
# # In[ ]:
cell_types = {}
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
marker = np.unique(col)

# In[ ]:
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


df.to_csv("new_taxonomy_table.csv")

# In[ ]:
markers_new = pd.read_csv("new_taxonomy_table.csv", index_col=0)

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
filen = "celltype_marker_table2.csv"

markers.to_csv(filen)
# In[ ]:
full_predictions, full_model = get_cell_types(raw_ad, markers)

# In[ ]:

filen = raw_data_path / XYLENA2_FULL_LABELS
full_predictions.reset_index(drop=True).to_feather(filen)
# In[ ]:
filen = raw_data_path / XYLENA2_FULL_CELLASSIGN
full_model.save(filen, overwrite=True)

del full_model
# In[ ]: collect metadata
ground_truth = pd.read_feather(
    raw_data_path / XYLENA2_FULL_LABELS
)  # XYLENA2_GROUND_TRUTH)
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
outfilen = data_path / XYLENA2_FULL

raw_ad.write_h5ad(outfilen)

# In[ ]:  make the train adata
train_ad = raw_ad[raw_ad.obs["train"]]

raw_train_filen = data_path / XYLENA2_TRAIN
train_ad.write_h5ad(raw_train_filen)
del train_ad

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
filen = "celltype_marker_table2.csv"
markers = pd.read_csv(filen, index_col=0)
ns_top_genes = [10_000, 5_000, 3_000, 2_000, 1_000]
ds_names = ["10k", "5k", "3k", "2k", "1k"]

marker_genes = markers.index.to_list()
# In[ ]:
full_filen = data_path / XYLENA2_FULL
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

hvgs_full.to_csv(data_path / XYLENA2_FULL_HVG)


# # In[ ]:
# train_filen = data_path / XYLENA2_TRAIN
# adata = ad.read_h5ad(train_filen)

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
# # process the "train" & "test" AnnData objects

# hvgs_train.to_csv(data_path / XYLENA2_TRAIN_HVG)

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
hvgs_full = pd.read_csv(data_path / XYLENA2_FULL_HVG, index_col=0)


## TODO: update to new marker genes tables....

# # In[ ]:
# # curate the gene list with our marker genes at the top.
# # markers = pd.read_csv("celltype_marker_table.csv", index_col=0)
# # mset = set(markers.index)
# # OLD
# markers = pd.read_csv("celltype_marker_table.csv", index_col=0)
# NEW
markers = pd.read_csv("celltype_marker_table2.csv", index_col=0)

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

gene_list.to_csv(data_path / XYLENA2_FULL_HVG)


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
def compute_pcs(
    adata: sc.AnnData, n_pcs: int = 50, save_path: Path | None = None, set_name="train"
) -> np.ndarray:
    """
    Compute principal components.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_pcs : int
        Number of principal components to compute.

    Returns
    -------
    pcs : ndarray
        Principal components.

    """
    bdata = adata.copy()
    print("compute_pcs - normalize_total")
    sc.pp.normalize_total(bdata, target_sum=1e4)
    print("compute_pcs - log1p")
    sc.pp.log1p(bdata)
    print(f"compute_pcs - pca: n_comps={n_pcs}")
    sc.pp.pca(bdata, n_comps=n_pcs, use_highly_variable=False)
    print("extracting pcs")
    pcs = bdata.varm["PCs"].copy()
    X_pca = bdata.obsm["X_pca"].copy()

    if save_path is not None:
        dump_pcs(pcs, save_path)
        print(f"Saved PCs to {save_path}")
        pcs_name = f"X_pca_{set_name}.npy"
        dump_x_pca(X_pca, save_path, pcs_name=pcs_name)
        print(f"Saved X_pca to {save_path}")

        return
    else:
        return pcs


def transfer_pca(
    adata: sc.AnnData,
    pcs: np.ndarray,
) -> sc.AnnData:
    """
    Transfer principal components.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    pcs : ndarray
        Principal components (eigenvectors) of training set.

    Returns
    -------
    x_pca : ndarray
        projection of data onto PCs.

    """
    bdata = adata.copy()
    # scale values.
    print("compute_pcs - normalize_total")
    sc.pp.normalize_total(bdata, target_sum=1e4)
    print("compute_pcs - log1p")
    sc.pp.log1p(bdata)
    X = bdata.X
    del bdata
    # Calculate the mean of each column
    col_means = X.sum(axis=0)
    col_means /= X.shape[0]

    # now chunk the X to get the x_pca.
    chunk_size = 10_000
    n_chunks = X.shape[0] // chunk_size

    X_pca = np.zeros((X.shape[0], pcs.shape[1]))
    for chunk in range(n_chunks):
        start = chunk * chunk_size
        end = start + chunk_size
        X_pca[start:end] = (X[start:end] - col_means) @ pcs

    # now do the last chunk
    start = n_chunks * chunk_size
    X_pca[start:] = (X[start:] - col_means) @ pcs

    # # Subtract the column means from each column
    # # col_means is 1x3 matrix; we use np.array to ensure proper broadcasting
    # adjusted_X = X - csr_matrix(np.ones((X.shape[0], 1))) @ csr_matrix(col_means)

    #     del bdata
    #     x_pca = np.matmul(X, pcs)
    #     # x_pca = bdata.X @ pcs
    return X_pca


def dump_pcs(pcs: np.ndarray, pcs_path: Path):
    """
    Save principal components (eigenvectors) of data.

    Parameters
    ----------
    pcs : ndarray
        Principal components.
    pcs_path : Path
        Path to save the PCs.


    """

    pcs_path = pcs_path / f"PCs.npy"
    np.save(pcs_path, pcs)


def dump_x_pca(x_pca: np.ndarray, pcs_path: Path, pcs_name: str = "X_pca.npy"):
    """
    Save principal components representation of data.

    Parameters
    ----------
    x_pca : ndarray
        Principal components.
    pcs_path : Path
        Path to save the PCs.


    """
    pcs_path = pcs_path / pcs_name
    np.save(pcs_path, x_pca)


def load_pcs(pcs_path: Path) -> np.ndarray:
    """
    Load principal components from adata.

    Parameters
    ----------
    pcs_path : Path
        Path to save the PCs.

    Returns
    -------
    pcs : np.ndarray
        Principal components.

    """

    pcs_path = pcs_path / f"PCs.npy"
    if pcs_path.exists():
        return np.load(pcs_path)
    else:
        print(f"no PCs found at {pcs_path}")
        return None


def load_x_pca(pcs_path: Path) -> np.ndarray:
    """
    Load principal components from adata.

    Parameters
    ----------
    pcs_path : Path
        Path to save the PCs.

    Returns
    -------
    X_pca : np.ndarray
        Principal components.

    """

    pcs_path = pcs_path / f"X_pca.npy"
    if pcs_path.exists():
        return np.load(pcs_path)
    else:
        print(f"no X_pca found at {pcs_path}")
        return None


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

    compute_pcs(train_ad, save_path=ds_path, set_name="train")

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

    # get loadings
    pcs = load_pcs(ds_path)
    X_pca = transfer_pca(test_ad, pcs)

    dump_x_pca(X_pca, ds_path, pcs_name=f"X_pca_test.npy")

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

    # get loadings
    pcs = load_pcs(ds_path)
    X_pca = transfer_pca(query_ad, pcs)

    dump_x_pca(X_pca, ds_path, pcs_name=f"X_pca_query.npy")

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
