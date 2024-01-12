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
    export_ouput_adata,
    make_scvi_normalized_adata,
)
from lbl8r import (
    get_lbl8r_scvi,
    get_lbl8r,
    prep_lbl8r_adata,
    query_lbl8r,
    query_scvi,
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

if __name__ == "__main__":
    save = True
    fdir = "figs"
    show = False
else:
    save = False
    fdir = None
    show = True


# In[ ]:
out_data_path = data_path / "LBL8R_scvi"

train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST

# In[ ]:
model_dir = "SCVI_nobatch"
cell_type_key = CELL_TYPE_KEY

# ### scVI model definition
# In[ ]:
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

retrain = True
plot_training = True


# In[ ]:
# ## 0. Load training data
train_ad = ad.read_h5ad(train_filen)


# In[ ]: Call the scvi model "vae" so we don't collide with scvi module
vae_model_name = "scvi"
vae, train_ad = get_lbl8r_scvi(  # sa,e ast get_trained_scvi but forces "batch"=None
    train_ad,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=vae_model_name,
    plot_training=plot_training,
    **fig_kwargs,
)

# In[ ]:
latent_ad = prep_lbl8r_adata(train_ad, vae, labels_key=cell_type_key)

# In[ ]:
lbl8r_model_name = "lbl8r"
retrain = True

labelator, latent_ad = get_lbl8r(
    latent_ad,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=lbl8r_model_name,
    plot_training=plot_training,
    **fig_kwargs,
)

# In[ ]:
# ## 3: visualize prediction fidelity on training set
plot_predictions(
    latent_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=lbl8r_model_name,
    title_str="TRAIN",
    **fig_kwargs,
)

# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(
    train_ad,
    basis=SCVI_MDE_KEY,
    color=[cell_type_key, "batch"],
    device=device,
    scvi_model=vae,
    **fig_kwargs,
)

# In[ ]:
# ------------------
# ## TEST
# ## 4.  Load data
test_ad = ad.read_h5ad(test_filen)

# In[ ]:
# ## 5 - prep lbl8r adata and query (run) model
latent_test_ad = prep_lbl8r_adata(test_ad, vae, labels_key=cell_type_key)
latent_test_ad = query_lbl8r(
    latent_test_ad,
    labelator,
    labels_key=cell_type_key,
)

# In[ ]:
# ## 6.  check the results on out-of-sample data
# - plot_predictions
# - visualize embeddings
plot_predictions(
    latent_test_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=lbl8r_model_name,
    title_str="TEST",
    **fig_kwargs,
)


# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(
    latent_test_ad,
    basis=SCVI_MDE_KEY,
    color=[cell_type_key, "batch"],
    device=device,
    scvi_model=vae,
    **fig_kwargs,
)

# In[ ]:
# ## 7: save versions of test/train with latents and embeddings added
export_ouput_adata(latent_ad, train_filen.name.replace(RAW, EMB), out_data_path)
export_ouput_adata(latent_test_ad, test_filen.name.replace(RAW, EMB), out_data_path)


# In[ ]:
# scvi_query, and vae should give the same results
exp_train_ad = make_scvi_normalized_adata(vae, train_ad)

# # In[ ]:
# # test that vae and scvi_query give the same results
# vaeexp_train_ad = make_scvi_normalized_adata(vae, train_ad)
# from numpy import allclose
# assert allclose(exp_train_ad.X, vaeexp_train_ad.X)


# In[ ]:
export_ouput_adata(exp_train_ad, train_filen.name.replace(RAW, EXPR), out_data_path)


# In[ ]:
del exp_train_ad, train_ad

exp_test_ad = make_scvi_normalized_adata(vae, test_ad)

# In[ ]:
export_ouput_adata(exp_test_ad, test_filen.name.replace(RAW, EXPR), out_data_path)
test_ad, exp_test_ad


# %%
del exp_test_ad

# In[ ]:
# _______________
# --------------
# ## make scVI normalized adata for further testing... i.e. `pcaLBL8R`
#
# > need to fit a SCVI_query model to get the expressions for the test data (which wasn't nescessary for the rest of the labelators)
#
# - Load the `vae` ("SCVI_nobatch").
# - transform the counts into expression
# - make the new AnnData
# - save
# In[ ]:
qscvi_model_name = "query_scvi"
scvi_query, test_ad = query_scvi(
    test_ad,
    vae,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=qscvi_model_name,
    plot_training=plot_training,
    **fig_kwargs,
)

# %%
del vae, scvi_query


# %%  ADD the PCs
import scanpy as sc
from lbl8r.utils import transfer_pcs

# %%
out_data_path = data_path / "LBL8R_scvi"
train_expn_filen = out_data_path / train_filen.name.replace(RAW, EXPR + OUT)
train_ad = ad.read_h5ad(train_expn_filen)

# %%
# DO PCA
sc.pp.pca(train_ad)

# %%
test_exp_filen = out_data_path / test_filen.name.replace(RAW, EXPR + OUT)
test_ad = ad.read_h5ad(test_exp_filen)

# In[ ]: # now we need to copy the PCs to the test set and compute loadings.
test_ad = transfer_pcs(train_ad, test_ad)
# In[ ]:
export_ouput_adata(train_ad, train_expn_filen.name.replace(RAW, EXPR), out_data_path)
export_ouput_adata(test_ad, test_exp_filen.name.replace(RAW, EXPR), out_data_path)


# %%
