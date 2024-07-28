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

sys.path.append(os.path.abspath("/home/ergonyc/Projects/SingleCell/labelator/"))

from lbl8r.model.utils._data import transfer_pcs

from lbl8r._constants import *


# ## create train and query datasets.

## e.g. for xylena data
XYLENA_ANNDATA = "brain_atlas_anndata.h5ad"
XYLENA_TRAIN = XYLENA_ANNDATA.replace(H5, TRAIN + CNT + H5)
XYLENA_TEST = XYLENA_ANNDATA.replace(H5, TEST + CNT + H5)

XYLENA_TRAIN_SPARSE = XYLENA_TRAIN.replace(H5, SPARSE + H5)
XYLENA_TEST_SPARSE = XYLENA_TEST.replace(H5, SPARSE + H5)


XYLENA_PATH = "data/scdata/xylena"
XYLENA_RAW_PATH = "data/scdata/xylena_raw"

XYLENA_METADATA = "final_metadata.csv"
XYLENA_ANNDATA2 = "brain_atlas_anndata_updated.h5ad"


## load raw data.
# In[ ]:
root_path = Path("../../")
data_path = root_path / XYLENA_PATH
raw_data_path = root_path / XYLENA_RAW_PATH

# In[ ]: Load raw data
raw_filen = raw_data_path / XYLENA_ANNDATA
raw_ad = ad.read_h5ad(raw_filen)

"""
This could also be acomplished by loading multiple samples and concatenating them.

e.g. something like this
"""

if False:
    samples = pd.read_csv("samples.csv")
    path_to_samples = Path("path/to/samples")
    file_names = path_to_samples / samples["sample_name"].str.cat(".h5ad")

    adatas = {
        sample_name: scanpy.read_h5ad(
            path_to_samples / samples["sample_name"].str.cat(".h5ad")
        )
        for sample_name in samples["sample_name"]
    }

    adata = anndata.concat(
        merge="same", uns_merge="same", index_unique="_", adatas=adatas
    )


# In[ ]: collect metadata
metadat = pd.read_csv(raw_data_path / XYLENA_METADATA)
obs = raw_ad.obs.copy()

# In[ ]: get the train/test splits
clean_samples_path = raw_data_path / "Model Combinations - clean_samples_138.csv"
clean_samples = pd.read_csv(clean_samples_path)
# all_samples_path = "/content/drive/MyDrive/SingleCellModel/Model Combinations - all_samples_199.csv"
# all_samples = pd.read_csv(all_samples_path)
# dirty_samples_path = "/content/drive/MyDrive/SingleCellModel/Model Combinations - dirty_samples_61.csv"
# dirty_samples = pd.read_csv(dirty_samples_path)
test_samples_path = raw_data_path / "Model Combinations - testing_set_41.csv"
test_samples = pd.read_csv(test_samples_path)
train_samples_path = raw_data_path / "Model Combinations - training_set_98.csv"
train_samples = pd.read_csv(train_samples_path)

# In[ ]:
newmeta = obs.join(metadat.set_index("cells"), lsuffix="", rsuffix="_other")

newmeta["clean"] = [s in set(clean_samples["sample"]) for s in newmeta["sample"]]
newmeta["test"] = [s in set(test_samples["sample"]) for s in newmeta["sample"]]
newmeta["train"] = [s in set(train_samples["sample"]) for s in newmeta["sample"]]


# In[ ]:
### WARNING:
# Fix the missing `cell_type` by inferring label from the class logits columns
# in newmeta.  Not sure why these were missing.
na_rows = newmeta["cell_type"].isna()
logit_cols = ["ExN1", "InN2", "MG3", "Astro4", "Oligo5", "OPC6", "VC7"]
newmeta.loc[na_rows, "cell_type"] = (
    newmeta.loc[na_rows, logit_cols].idxmax(axis=1).str[:-1]
)

newmeta["long_type"] = (
    newmeta["cell_type"].astype(str) + " " + newmeta["type"].astype(str)
)


# In[ ]:
# update anndata - only keep the metadata we need/want
raw_ad.obs = newmeta["cell_type", "sample", "batch", "test", "train"]

# In[ ]: save updated full AnnData object with updated metadata (obs)
outfilen = raw_data_path / XYLENA_ANNDATA2
raw_ad.write_h5ad(outfilen)

# In[ ]:  make the tran and test anndatas
train_ad = raw_ad[raw_ad.obs["train"]].copy()
test_ad = raw_ad[raw_ad.obs["test"]].copy()

# In[ ]:clean up
del raw_ad
del metadat, obs

# In[ ]:
raw_train_filen = raw_data_path / XYLENA_ANNDATA.replace(H5, TRAIN + H5)
train_ad.write_h5ad(raw_train_filen)

raw_test_filen = raw_data_path / XYLENA_ANNDATA.replace(H5, TEST + H5)
test_ad.write_h5ad(raw_test_filen)

# In[ ]: delete to save memory and reload one at at time
del train_ad, test_ad

# In[ ]: load the raw train_ad
raw_train_filen = raw_data_path / XYLENA_ANNDATA.replace(H5, TRAIN + H5)
raw_test_filen = raw_data_path / XYLENA_ANNDATA.replace(H5, TEST + H5)

train_ad = ad.read_h5ad(raw_train_filen, backed=False)

# In[ ]:
# use default scanpy for the pca
sc.pp.pca(train_ad)

# In[ ]:
# export the train_ad with pcas
train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST

# In[ ]:
train_ad.write_h5ad(train_filen)

# In[ ]:
# load the raw test_ad
test_ad = ad.read_h5ad(raw_test_filen)

# In[ ]: # now we need to copy the PCs to the test set and compute loadings.
test_ad = transfer_pcs(test_ad, train_ad)
# In[ ]:
test_ad.write_h5ad(test_filen)

# In[ ]:
del test_ad
del train_ad

# Optional save memory/disk space by using sparse matrices.  Models will train / infer slower
# In[ ]:
train_ad = ad.read_h5ad(train_filen)

# also coudl use lbl8r.model.utils._data.sparsify_adata

train_ad.X = sp.csr_matrix(train_ad.X)
train_filen = data_path / XYLENA_TRAIN_SPARSE
train_ad.write_h5ad(train_filen)


# In[ ]:
test_ad = ad.read_h5ad(test_filen)

test_ad.X = sp.csr_matrix(test_ad.X)
test_filen = data_path / XYLENA_TEST_SPARSE
test_ad.write_h5ad(test_filen)


# In[ ]:
del test_ad, train_ad

# CNT / FULL
full_ad = ad.read_h5ad(raw_data_path / XYLENA_ANNDATA2)
full_ad.X = sp.csr_matrix(full_ad.X)

outfilen = raw_data_path / XYLENA_ANNDATA.replace(H5, SPARSE + H5)
full_ad.write_h5ad(outfilen)
del full_ad


# In[ ]:
