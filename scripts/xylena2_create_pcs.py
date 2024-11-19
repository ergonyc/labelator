#!/usr/bin/env python
# coding: utf-8

### set up train and test AnnData objects for LBL8R

# In[ ]:
### import local python functions in ../lbl8r
import sys

import scanpy as sc
import anndata as ad
from pathlib import Path
import scipy.sparse as sp
import pandas as pd
import numpy as np


sys.path.append(Path.cwd().parent.as_posix())

from lbl8r.model.utils._pca import transfer_pca, compute_pcs, dump_pcs, dump_x_pca

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

XYLENA2_RAW_PATH = "scdata/xylena_raw"
XYLENA2_PATH = "scdata/xylena"

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


# In[ ]:


ds_names = ["10k", "5k", "3k", "2k", "1k"]
ds_names = ["5k", "3k", "2k", "1k"]
ds_names = ["10k"]
for ds_name in ds_names:
    ds_path = data_path / ds_name

    ds_path = data_path / ds_name
    ds_path.mkdir(exist_ok=True)
    train_filen = ds_path / XYLENA2_TRAIN
    adata = ad.read_h5ad(train_filen)

    pcs = compute_pcs(adata)

    x_pca_name = f"X_pca_{train_filen.name.replace('h5ad', 'npy')}"

    pcs, X_pca = compute_pcs(adata)
    # save the PCs and X_pca

    dump_pcs(pcs, ds_path)
    dump_x_pca(X_pca, ds_path, xpca_name=x_pca_name)

    # test
    test_filen = ds_path / XYLENA2_TEST
    adata = ad.read_h5ad(test_filen)
    X_pca = transfer_pca(adata, pcs)
    x_pca_name = f"X_pca_{test_filen.name.replace('h5ad', 'npy')}"
    dump_x_pca(X_pca, ds_path, xpca_name=x_pca_name)

    # query
    query_filen = ds_path / XYLENA2_QUERY
    adata = ad.read_h5ad(test_filen)
    X_pca = transfer_pca(adata, pcs)
    x_pca_name = f"X_pca_{query_filen.name.replace('h5ad', 'npy')}"
    dump_x_pca(X_pca, ds_path, xpca_name=x_pca_name)

    del adata

# %%

# ds_names = ["10k", "5k", "3k", "2k", "1k"]
# ds_names = ["5k", "3k", "2k", "1k"]

# for ds_name in ds_names:
#     ds_path = data_path / ds_name
#     train_filen = ds_path / XYLENA2_TRAIN
#     adata = ad.read_h5ad(train_filen)

#     # get total number of genes
