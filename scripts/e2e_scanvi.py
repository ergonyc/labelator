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
sys.path.append(os.path.abspath((os.path.join(os.getcwd(), '..'))))

from lbl8r.utils import (
            plot_predictions,
            plot_embedding,
            export_ouput_adata,
            make_scvi_normalized_adata
            )
from lbl8r import (
                get_trained_scanvi, 
                get_trained_scvi, 
                query_scanvi, 
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
    fig_dir = "figs"
    show = False
else:
    save = False
    fig_dir = None
    show = True

# control figure saving and showing here
fig_kwargs = dict(
    save = save,
    show = show, 
    fig_dir = fig_dir,
)

# In[ ]:
out_path = data_path / "SCANVI"

# ## 0. Load training data
train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST

# In[ ]:
# setup anndata & covariate keys.
model_dir = "SCANVI"
cell_type_key = "cell_type"

# In[ ]:
model_root_path = root_path / MODEL_SAVE_DIR
if not model_root_path.exists():
    model_root_path.mkdir()

model_path = model_root_path / model_dir
if not model_path.exists():
    model_path.mkdir()

if fig_dir is not None:
    fig_dir = Path(fig_dir) / model_dir
    if not fig_dir.exists():
        fig_dir.mkdir()
    
retrain=False
plot_training = True

# In[ ]: LOAD TRAIN DATA
train_ad = ad.read_h5ad(train_filen)


# In[ ]: for scanvi we need to query with "unknown" cell_types so save "ground_truth" before overwriting
train_ad.obs["ground_truth"] = train_ad.obs[cell_type_key]
# ### Model setup
batch_key = "sample"  #'batch'
vae_model_name = "scvi"
vae, train_ad = get_trained_scvi(
    train_ad,
    labels_key=cell_type_key,
    batch_key=batch_key,
    model_path=model_path,
    retrain=retrain,
    model_name=vae_model_name,
    plot_training=True,
    **fig_kwargs,
)

# In[ ]: Now we can train scANVI and transfer the labels!
scanvi_model_name = "scanvi"
scanvi_model, train_ad = get_trained_scanvi(
    train_ad,
    vae,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=scanvi_model_name,
    plot_training=True,
    **fig_kwargs,
)

# In[ ]:
plot_predictions(
            train_ad,
            pred_key=SCANVI_PREDICTIONS_KEY,
            cell_type_key=cell_type_key,
            model_name=scanvi_model_name,
            title_str="TRAIN",
            save = save,
            show = show, 
            fig_dir = fig_dir,
        )


# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(
    train_ad,
    basis="X_scVI_mde",
    color=["C_scANVI", "batch"],
    frameon=False,
    wspace=0.35,
    device=device,
    scvi_model=vae,
    **fig_kwargs,

)

# In[ ]:
export_ouput_adata(train_ad, train_filen.name, out_path)

# In[ ]:
# ----------------------
# ## Test & Probe
test_ad = ad.read_h5ad(test_filen)

# "Hide" the labels
test_ad.obs["ground_truth"] = test_ad.obs[cell_type_key]
# pretend like we don't know the cell_type
test_ad.obs[cell_type_key] = "Unknown"
# # we can't just project the test data onto the reference model, because the batch variables are unknown
# scvi_lat = scvi_ref.get_latent_representation(test_ad)

# In[ ]:
# ## "Inference"
# ### query scVI model with test data

qscvi_model_name = "query_scvi"
scvi_query, test_ad = query_scvi(
    test_ad,
    vae,
    labels_key=cell_type_key,
    model_path=model_path,
    model_name=qscvi_model_name,
    retrain=retrain,
    plot_training=plot_training,
    **fig_kwargs,
)

retrain = True
# In[ ]:
# ### query scANVI model with test data
qscanvi_model_name = "query_scanvi"
scanvi_query, test_ad = query_scanvi(
    test_ad,
    scanvi_model,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=qscanvi_model_name,
    plot_training=True,
    **fig_kwargs,
)
# In[ ]:
# ## Assessment 
# Compute Accuracy of model classifier for query dataset and compare predicted 
# and observed cell types
plot_predictions(
    test_ad,
    pred_key=SCANVI_PREDICTIONS_KEY,
    cell_type_key="ground_truth",
    model_name=qscanvi_model_name,
    title_str="TEST",
    **fig_kwargs,
)

# In[ ]:
# ### save train and test adatas with embeddings 
# We have already added the `scVI` and `scANVI` embeddings to the obsm
# Lets also compute the PCAs and MSE embeddings for visualization.

# this should also add the embeddings to the adata
plot_embedding(
    test_ad,
    basis="X_scVI_mde",
    color=[SCANVI_PREDICTIONS_KEY, "batch"],
    frameon=False,
    wspace=0.35,
    device=device,
    scvi_model=scvi_query,
    **fig_kwargs,
)

# In[ ]:
# reset the cell_type_key before exporting
test_ad.obs[cell_type_key] = test_ad.obs["ground_truth"]

export_ouput_adata(test_ad, test_filen.name, out_path) # will append "_out.h5ad"

# In[ ]:
## reload the saved adatas and make the scvi normalized adata for further testing... i.e. `pcaLBL8R`
# # _______________
# ## make scVI normalized adata for further testing... i.e. `pcaLBL8R`
# 
# - Load the `vae` ("SCVI").  
# - transform the counts into expression
# - make the new AnnData
# - save
train_ad = ad.read_h5ad( out_path / train_filen.name.replace(".h5ad","_out.h5ad"))
exp_train_ad = make_scvi_normalized_adata(scvi_query, train_ad)

# In[ ]:
export_ouput_adata(exp_train_ad, train_filen.name.replace("_cnt.h5ad", "_exp.h5ad"), out_path)
del exp_train_ad, train_ad

# In[ ]:
# reset the cell_type_key before exporting
test_ad.obs[cell_type_key] = test_ad.obs["ground_truth"]
exp_test_ad = make_scvi_normalized_adata(scvi_query, test_ad)
export_ouput_adata(exp_test_ad, test_filen.name.replace("_cnt.h5ad", "_exp.h5ad"), out_path)

