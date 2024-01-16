#!/usr/bin/env python
# coding: utf-8
# In[ ]:
# imports
import sys
import os

from pathlib import Path
import scanpy as sc
import torch
import anndata as ad
import scvi

### import local python functions in ../lbl8r
sys.path.append(os.path.abspath((os.path.join(os.getcwd(), ".."))))

from lbl8r.utils import (
    plot_predictions,
    plot_embedding,
    plot_lbl8r_training,
    export_ouput_adata,
)
from lbl8r import (
    prep_lbl8r_adata,
    get_pca_lbl8r,
    query_lbl8r,
)
from lbl8r.constants import *
from lbl8r.constants import XYLENA_PATH

torch.set_float32_matmul_precision("medium")
sc.set_figure_params(figsize=(4, 4))
scvi.settings.seed = 94705

device = "mps" if sys.platform == "darwin" else "cuda"

# In[ ]:
root_path = Path("../")

data_path = root_path / XYLENA_PATH

if "ipykernel" in sys.modules:
    save = True
    fdir = "figs"
    show = False
else:
    save = False
    fdir = None
    show = True

# In[ ]:
out_data_path = data_path / f"LBL8R{PCS}"

train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST

# In[ ]:
model_dir = "PCS"
cell_type_key = CELL_TYPE_KEY


# In[ ]: ## model definition
# Here we want to classify based on the PCA loadings.
# Hand define a helper multilayer perceptron class to use it with a VAE below.
model_root_path = root_path / MODEL_SAVE_DIR
if not model_root_path.exists():
    model_root_path.mkdir()

model_path = model_root_path / model_dir
if not model_path.exists():
    model_path.mkdir()

if fdir is not None:
    fig_dir = Path(fdir) / model_dir
    if not fig_dir.exists():
        fig_dir.mkdir()

# control figure saving and showing here
fig_kwargs = dict(
    save=save,
    show=show,
    fig_dir=fig_dir,
)

retrain = False
plot_training = True
pca_model_name = "lbl8r_pcs"

# In[ ]:
# TRAIN #######################################
############################################################################
## LOAD  ###################################################################
train_ad = ad.read_h5ad(train_filen)
pca_train_ad = prep_lbl8r_adata(train_ad, pca_key=PCA_KEY, labels_key=cell_type_key)

## GET   ###################################################################
labelator, train_ad = get_pca_lbl8r(  # get_lbl8r
    pca_train_ad,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=pca_model_name,
    plot_training=plot_training,
    **fig_kwargs,
)

train_ad = query_lbl8r(
    train_ad,
    labelator,
    labels_key=cell_type_key,
)


# In[ ]:
# TEST #######################################
############################################################################
## LOAD  ###################################################################
test_ad = ad.read_h5ad(test_filen)


# In[ ]:
# ## 5 - prep lbl8r adata and query (run) model
pca_test_ad = prep_lbl8r_adata(test_ad, pca_key=PCA_KEY, labels_key=cell_type_key)

## QUERY  ###################################################################
pca_test_ad = query_lbl8r(
    pca_test_ad,
    labelator,
    labels_key=cell_type_key,
)
# ARTIFACTS ###########################################################################
############################################################################
## PLOTS  ###################################################################
# ## 3: visualize prediction fidelity on training set

# PLOT predictions ###############################################################
############################################################################
plot_predictions(
    train_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=pca_model_name,
    title_str="TRAIN",
    **fig_kwargs,
)

# In[ ]:
plot_predictions(
    test_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=pca_model_name,
    title_str="TEST",
    **fig_kwargs,
)
# PLOT embeddings ###############################################################
############################################################################
# this should also add the embeddings to the adata
plot_embedding(
    train_ad,
    basis=MDE_KEY,
    color=[cell_type_key, "batch"],
    device=device,
    **fig_kwargs,
)

plot_embedding(
    test_ad,
    basis=MDE_KEY,
    color=[cell_type_key, "batch"],
    device=device,
    **fig_kwargs,
)

# PLOT TRAINING ###############################################################
############################################################################
if plot_training:
    plot_lbl8r_training(labelator.history, save=save, show=show, fig_dir=fig_dir)


# In[ ]:
# ## 7: save versions of test/train with latents and embeddings added
export_ouput_adata(pca_train_ad, train_filen.name.replace(H5, PCS + H5), out_data_path)
export_ouput_adata(pca_test_ad, test_filen.name.replace(H5, PCS + H5), out_data_path)
