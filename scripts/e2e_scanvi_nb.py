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
    plot_scanvi_training,
    plot_scvi_training,
    export_ouput_adata,
    make_scvi_normalized_adata,
    transfer_pcs,
)
from lbl8r import (
    get_trained_scanvi,
    get_trained_scvi,
    get_query_scanvi,
    get_query_scvi,
    query_scanvi,
)
from lbl8r.constants import *
from lbl8r.constants import XYLENA_PATH

torch.set_float32_matmul_precision("medium")
sc.set_figure_params(figsize=(4, 4))
scvi.settings.seed = 94705

device = "mps" if sys.platform == "darwin" else "cuda"

# In[ ]:
# setup 1 #######################################
############################################################################
############################################################################
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
out_data_path = data_path / "SCANVI_nobatch"

# ## 0. Load training data
train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST

# In[ ]:
# setup anndata & covariate keys.
model_dir = "SCANVI_nobatch"
cell_type_key = CELL_TYPE_KEY

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

batch_key = None  #'batch'
vae_model_name = "scvi"
scanvi_model_name = "scanvi"
qscvi_model_name = "query_scvi"
qscanvi_model_name = "query_scanvi"

# In[ ]: LOAD TRAIN DATA
# In[ ]: LOAD TRAIN DATA
# TRAIN #######################################
############################################################################
## LOAD  ###################################################################
train_ad = ad.read_h5ad(train_filen)
train_ad.obs["ground_truth"] = train_ad.obs[cell_type_key]


# In[ ]: for scanvi we need to query with "unknown" cell_types so save "ground_truth" before overwriting
# ### Model setup
vae, train_ad = get_trained_scvi(
    train_ad,
    labels_key=cell_type_key,
    batch_key=batch_key,
    model_path=model_path,
    retrain=retrain,
    model_name=vae_model_name,
    **fig_kwargs,
)

# In[ ]: Now we can train scANVI and transfer the labels!
scanvi_model, train_ad = get_trained_scanvi(
    train_ad,
    vae,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=scanvi_model_name,
    **fig_kwargs,
)

# In[ ]:
# TEST #######################################
############################################################################
## LOAD  ###################################################################

test_ad = ad.read_h5ad(test_filen)

# "Hide" the labels
test_ad.obs["ground_truth"] = test_ad.obs[cell_type_key]
# pretend like we don't know the cell_type
test_ad.obs[cell_type_key] = "Unknown"
# # we can't just project the test data onto the reference model, because the batch variables are unknown
# scvi_lat = scvi_ref.get_latent_representation(test_ad)

# In[ ]: "Inference" query scVI model with test data
## QUERY  ###################################################################
############################################################################
scvi_query, test_ad = get_query_scvi(
    test_ad,
    vae,
    labels_key=cell_type_key,
    model_path=model_path,
    model_name=qscvi_model_name,
    retrain=retrain,
    plot_training=plot_training,
    **fig_kwargs,
)

# In[ ]:
# ### query scANVI model with test data
scanvi_query, test_ad = get_query_scanvi(
    test_ad,
    scanvi_model,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=qscanvi_model_name,
    plot_training=plot_training,
    **fig_kwargs,
)

## QUERY  ###################################################################
train_ad = query_scanvi(train_ad, scanvi_model, insert_key=SCANVI_PREDICTIONS_KEY)
test_ad = query_scanvi(test_ad, scanvi_query, insert_key=SCANVI_PREDICTIONS_KEY)


# In[ ]:
# ARTIFACTS ###########################################################################
############################################################################
## PLOTS  ###################################################################
# ## 3: visualize prediction fidelity on training set

# PLOT predictions ###############################################################
############################################################################
plot_predictions(
    test_ad,
    pred_key=SCANVI_PREDICTIONS_KEY,
    cell_type_key="ground_truth",
    model_name=qscanvi_model_name,
    title_str="TEST",
    **fig_kwargs,
)

# In[ ]:
plot_predictions(
    train_ad,
    pred_key=SCANVI_PREDICTIONS_KEY,
    cell_type_key=cell_type_key,
    model_name=scanvi_model_name,
    title_str="TRAIN",
    **fig_kwargs,
)


# In[ ]:
# PLOT embeddings ###############################################################
############################################################################
plot_embedding(
    test_ad,
    basis=SCVI_MDE_KEY,
    color=[SCANVI_PREDICTIONS_KEY, "batch"],
    frameon=False,
    wspace=0.35,
    device=device,
    scvi_model=scvi_query,
    **fig_kwargs,
)
# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(
    train_ad,
    basis=SCVI_MDE_KEY,
    color=["C_scANVI", "batch"],
    device=device,
    scvi_model=vae,
    **fig_kwargs,
)

# PLOT TRAINING ###############################################################
############################################################################
if plot_training:
    plot_scvi_training(
        vae.history, save=save, show=show, fig_dir=fig_dir
    )  #    **fig_kwargs

if plot_training:
    plot_scanvi_training(scanvi_model.history, save=save, show=show, fig_dir=fig_dir)

if plot_training:
    plot_scvi_training(scvi_query.history, save=save, show=show, fig_dir=fig_dir)

# In[ ]:
## ADATAS  ###################################################################
############################################################################
export_ouput_adata(train_ad, train_filen.name, out_data_path)


# reset the cell_type_key before exporting
test_ad.obs[cell_type_key] = test_ad.obs["ground_truth"]
export_ouput_adata(test_ad, test_filen.name, out_data_path)  # will append "_out.h5ad"


############################################################################
############################################################################
############################################################################
############################################################################


# In[ ]:
## reload the saved adatas and make the scvi normalized adata for further testing...
# # _______________
# ## make scVI normalized adata for further testing... i.e. `pcaLBL8R`
#
# - Load the `vae` ("SCVI").
# - transform the counts into expression
# - make the new AnnData
# - save
train_ad = ad.read_h5ad(out_data_path / train_filen.name.replace(H5, OUT + H5))
exp_train_ad = make_scvi_normalized_adata(scvi_query, train_ad)

# In[ ]:
export_ouput_adata(exp_train_ad, train_filen.name.replace(RAW, EXPR), out_data_path)
del exp_train_ad, train_ad

test_ad = ad.read_h5ad(out_data_path / test_filen.name.replace(H5, OUT + H5))

# In[ ]:
# reset the cell_type_key before exporting
test_ad.obs[cell_type_key] = test_ad.obs["ground_truth"]
exp_test_ad = make_scvi_normalized_adata(scvi_query, test_ad)
export_ouput_adata(exp_test_ad, test_filen.name.replace(RAW, EXPR), out_data_path)

# In[ ]:
## reload the saved adatas and make the scvi normalized adata for further testing... i.e. `pcaLBL8R`
# # _______________
# ## make scVI normalized adata for further testing... i.e. `pcaLBL8R`
#
# - Load the `vae` ("SCVI").
# - transform the counts into expression
# - make the new AnnData
# - save

train_ad = ad.read_h5ad(out_data_path / train_filen.name.replace(H5, OUT + H5))
train_ad = make_scvi_normalized_adata(scvi_query, train_ad)


# In[ ]:
# reset the cell_type_key before exporting
test_ad.obs[cell_type_key] = test_ad.obs["ground_truth"]
test_ad = make_scvi_normalized_adata(scvi_query, test_ad)


# In[ ]:
# pcs
sc.pp.pca(train_ad)
test_ad = transfer_pcs(train_ad, test_ad)

# In[ ]:
export_ouput_adata(train_ad, train_filen.name.replace(RAW, EXPR), out_data_path)
export_ouput_adata(test_ad, test_filen.name.replace(RAW, EXPR), out_data_path)


# TODO:  make minified version of the adata for testing
