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
out_data_path = data_path / "LBL8R"+EMB

train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST

# In[ ]:
# ## 0. Load training data

train_ad = ad.read_h5ad(train_filen)

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

if fig_dir is not None:
    fig_dir = Path(fig_dir) / model_dir
    if not fig_dir.exists():
        fig_dir.mkdir()
    
retrain = False
plot_training = True


# In[ ]:
vae_model_name = "scvi_nobatch"
vae, train_ad = get_lbl8r_scvi( # sa,e ast get_trained_scvi but forces "batch"=None
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
retrain = False

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
plot_embedding(train_ad,
               basis=SCVI_MDE_KEY,
                color=[cell_type_key, "batch"],
                device=device,
                scvi_model = vae,
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
plot_predictions(latent_test_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=lbl8r_model_name, 
                 title_str="TEST",
                 **fig_kwargs,

            )


# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(latent_test_ad,
               basis=SCVI_MDE_KEY,
                color=[cell_type_key, "batch"],
                device=device,
                scvi_model = vae,
                **fig_kwargs,
            )

# In[ ]:
# ## 7: save versions of test/train with latents and embeddings added
export_ouput_adata(latent_ad, train_filen.name.replace(RAW, EMB+NOBATCH), out_data_path)
export_ouput_adata(latent_test_ad, test_filen.name.replace(RAW, EMB+NOBATCH), out_data_path)


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


# In[ ]:
# scvi_query, and vae should give the same results
exp_train_ad = make_scvi_normalized_adata(scvi_query, train_ad)


# In[ ]:
export_ouput_adata(exp_train_ad, train_filen.name.replace(RAW, EXPR+NOBATCH), out_data_path)


# In[ ]:
del exp_train_ad, train_ad

exp_test_ad = make_scvi_normalized_adata(scvi_query, test_ad)
export_ouput_adata(exp_test_ad, test_filen.name.replace(RAW, EXPR+NOBATCH), out_data_path)
del exp_test_ad, test_ad

