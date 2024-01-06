#!/usr/bin/env python
# coding: utf-8

# ## Prototype LABELATOR with anndata pytorch loader
#

import sys

from pathlib import Path

import scanpy as sc
import torch

import matplotlib.pyplot as plt
import numpy as np
import anndata as ad

from scvi.model import SCVI


from lbl8r.utils import (
    make_latent_adata,
    add_predictions_to_adata,
    plot_predictions,
    plot_embedding,
    export_ouput_adata,
    make_scvi_normalized_adata,
    make_pc_loading_adata,
)

from lbl8r.lbl8r import scviLBL8R
from lbl8r import (
    get_lbl8r_scvi,
    add_lbl8r_classifier,
    prep_lbl8r_adata,
    query_lbl8r,
    query_scvi,
    get_pca_lbl8r,
)


sc.set_figure_params(figsize=(4, 4))


# Stubs to find the data

root_path = Path("../")
data_path = root_path / "data/scdata/xylena"
raw_data_path = root_path / "data/scdata/xylena_raw"

XYLENA_ANNDATA = "brain_atlas_anndata.h5ad"
XYLENA_METADATA = "final_metadata.csv"
XYLENA_ANNDATA2 = "brain_atlas_anndata_updated.h5ad"

XYLENA_TRAIN = XYLENA_ANNDATA.replace(".h5ad", "_train_cnt.h5ad")
XYLENA_TEST = XYLENA_ANNDATA.replace(".h5ad", "_test_cnt.h5ad")

XYLENA_TRAIN_SPARSE = XYLENA_TRAIN.replace(".h5ad", "_sparse.h5ad")
XYLENA_TEST_SPARSE = XYLENA_TEST.replace(".h5ad", "_sparse.h5ad")


# # subsample 40k cells for prototyping
# subsamples = np.random.choice(train_ad.shape[0], 40_000, replace=False)
# train_ad = train_ad[subsamples,:].copy() # no copy... just alias


# In[6]:


CELL_TYPE_KEY = "cell_type"
OUT_PATH = data_path / "LBL8R"


# ## 0. Load training data

# In[7]:


filen = data_path / XYLENA_TRAIN
train_ad = ad.read_h5ad(filen)


# ## model definition

# Here we want to classify based on the PCA loadings.

# Here we define a helper multilayer perceptron class to use it with a VAE below.

# In[8]:


model_path = root_path / "lbl8r_models"
if not model_path.exists():
    model_path.mkdir()

retrain = False


#
# ## 1 setup data and load `pcaLBL8R`
#
# `setup_anndata` and load model.  (Or instantiate and train)
#

# In[9]:


MODEL_NAME = "LBL8R_pca"
labelator, train_ad = get_pca_lbl8r(
    train_ad,
    labels_key=CELL_TYPE_KEY,
    model_path=model_path,
    retrain=retrain,
    model_name=MODEL_NAME,
    plot_training=True,
)


# In[10]:


plot_predictions(
    train_ad,
    pred_key="pred",
    cell_type_key=CELL_TYPE_KEY,
    model_name=MODEL_NAME,
    title_str="TRAIN",
)


# In[11]:


# this should also add the embeddings to the adata
plot_embedding(
    train_ad,
    basis="X_mde",
    color=[CELL_TYPE_KEY, "batch"],
    frameon=False,
    wspace=0.35,
    device="cuda",
)


# ------------------
# Now TEST

# In[12]:


filen = data_path / XYLENA_TEST
test_ad = ad.read_h5ad(filen)

# In[13]:


test_ad = make_pc_loading_adata(test_ad)

#
# latent_test_ad = prep_lbl8r_adata(test_ad, vae, labels_key=CELL_TYPE_KEY)


# In[14]:


test_ad = query_lbl8r(
    test_ad,
    labelator,
    labels_key=CELL_TYPE_KEY,
)


# In[15]:


plot_predictions(
    test_ad,
    pred_key="pred",
    cell_type_key=CELL_TYPE_KEY,
    model_name="pca_lbl8r",
    title_str="TEST",
)


# this should also add the embeddings to the adata
plot_embedding(
    test_ad,
    basis="X_mde",
    color=[CELL_TYPE_KEY, "batch"],
    frameon=False,
    wspace=0.35,
    device="cuda",
)


# ## 7: save versions of test/train with latents and embeddings added

# In[17]:
export_ouput_adata(train_ad, XYLENA_TRAIN.replace("_cnt.h5ad", "_pca.h5ad"), OUT_PATH)

export_ouput_adata(test_ad, XYLENA_TEST.replace("_cnt.h5ad", "_pca.h5ad"), OUT_PATH)
