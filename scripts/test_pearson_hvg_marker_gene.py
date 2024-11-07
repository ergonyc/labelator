### test_pearson_residuals + marker genes

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

from lbl8r._constants import *

# In[ ]:
## e.g. for xylena data
XYLENA2_RAW_ANNDATA = "full_object.h5ad"
XYLENA2_GROUND_TRUTH = "ground_truth_labels.csv"

XYLENA2_FULL = "xyl2_full.h5ad"

XYLENA2_TRAIN = "xyl2_train.h5ad"
XYLENA2_TEST = "xyl2_test.h5ad"
XYLENA2_QUERY_A = "xyl2_query.h5ad"
XYLENA2_QUERY_B = "xyl2_queryB.h5ad"

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
########################
# 2. HIGHLY VARIABLE GENES
########################
# ns_top_genes = [20_000, 10_000, 5_000, 3_000, 2_000, 1_000]
# ds_names = ["20k", "10k", "5k", "3k", "2k", "1k"]

ns_top_genes = [10_000, 5_000, 3_000, 2_000, 1_000]
ds_names = ["10k", "5k", "3k", "2k", "1k"]

# In[ ]:
full_filen = data_path / XYLENA2_TEST #XYLENA2_FULL
# adata = ad.read_h5ad(full_filen,backed='r+')
adata = ad.read_h5ad(full_filen)

# # In[ ]:
# hvgs_full = sc.experimental.pp.highly_variable_genes(
#     adata,
#     n_top_genes=20_000,
#     batch_key="sample",
#     flavor="pearson_residuals",
#     check_values=True,
#     layer=None,
#     subset=False,
#     inplace=False,
# )

# In[ ]:
# hvgs_train = pd.read_csv(raw_data_path / XYLENA2_TRAIN_HVG, index_col=0)
hvgs_full = pd.read_csv(data_path / XYLENA2_FULL_HVG, index_col=0)



# In[ ]:
filen = "../../brain-taxonomy/markers/cellassign_card_markers.csv"
markers = pd.read_csv(filen, index_col=0)

# In[ ]:
markers.head()
# hvgs = set(hvgs_full.index)
# In[ ]:
# hvgs_full.loc[markers.index, 'highly_variable_rank'] = 1.
hvgs_full.loc[markers.index, "highly_variable_nbatches"] = hvgs_full['highly_variable_nbatches'].max()+1.0

# In[ ]:
# Sort genes by how often they selected as hvg within each batch and
# break ties with median rank of residual variance across batches
hvgs_full.sort_values(
    ["highly_variable_nbatches", "highly_variable_rank"],
    ascending=[False, True],
    na_position="last",
    inplace=True,
)
# %%
gene_list = hvgs_full.copy()

gene_list["marker"] = False
gene_list.loc[markers.index, "marker"] = True


# In[ ]:

n_top_genes = 3_000
keep_genes= gene_list.iloc[:n_top_genes].index.to_list()

# In[ ]:

adata1 = adata[:, keep_genes]
adata2 = adata[:, adata.var.index.isin(keep_genes)]

# %%
