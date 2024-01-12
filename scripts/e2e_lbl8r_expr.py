#!/usr/bin/env python
# coding: utf-8

# ## Prototype end to end Labelator E2E_LBL8R 
# ### List of models
# 
# e2e MLP classifier variants:
# - raw counts: n=3000 features
# - normalized counts (scVI)
# 
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
            add_predictions_to_adata,
            make_scvi_normalized_adata
            )
from lbl8r import (
                get_lbl8r,  
                query_lbl8r,
                get_lbl8r_scvi,
                )

from lbl8r.models import LBL8R
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


# In[8]:
# ## `AnnData` initialization
out_data_path = data_path / f"LBL8R{RAW}"

# ## 0. training data
train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST


# In[9]:

# ## model definition
# Here we want to classify based on the raw counts
# Here we define a helper multilayer perceptron class to use it with a VAE below.

model_dir = "E2E_LBL8R"
cell_type_key = CELL_TYPE_KEY

# In[6]:
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
    
fig_kwargs = dict(
    save = save,
    show = show, 
    fig_dir = fig_dir,
)

retrain=True
plot_training = True


# In[ ]: LOAD TRAIN DATA
train_ad = ad.read_h5ad(train_filen)

# In[ ]:
lbl8r_model_name = "raw_cnt"
labelator, train_ad = get_lbl8r( 
    train_ad,
    labels_key=cell_type_key,
    model_path=model_path,
    model_name=lbl8r_model_name,
    retrain=retrain,
    plot_training=plot_training,
    **fig_kwargs,
)

# In[ ]:
# ## 3: visualize prediction fidelity on training set
plot_predictions(
    train_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=lbl8r_model_name,
    title_str="TRAIN",
    **fig_kwargs,
    )

# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(train_ad,
               basis=MDE_KEY,
                color=[cell_type_key, "batch"],
                device=device,
                **fig_kwargs,
                )

# save versions of test/train with latents and embeddings added
# 
# In[17]:
# train_ad.write_h5ad(data_path / train_filen.name.replace(".h5ad", "_out.h5ad") )
export_ouput_adata(train_ad, train_filen.name, out_data_path)

# In[ ]:
# ------------------
# ## TEST
# ## 4.  Load data
test_ad = ad.read_h5ad(test_filen)

# test_ad.obs["ground_truth"] = test_ad.obs[cell_type_key]

# In[19]:

test_ad = query_lbl8r(
    test_ad,
    labelator,
    labels_key=cell_type_key,
)


# LBL8R.setup_anndata(test_ad, labels_key=cell_type_key)

# test_predictions = labelator.predict(test_ad, probs=False, soft=True)
# # In[20]:
# test_ad = add_predictions_to_adata(
#     test_ad, test_predictions, insert_key="pred", pred_key="label"
# )
# In[ ]:
plot_predictions(
    test_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=lbl8r_model_name,
    title_str="TEST",
    **fig_kwargs,

)

# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(test_ad,
               basis=PCA_KEY,
                color=[cell_type_key, "batch"],
                device=device,
                **fig_kwargs,
            )
# save versions of test/train with latents and embeddings added
# 
# In[23]:
# test_ad.write_h5ad(data_path / test_filen.name.replace(".h5ad", "_out.h5ad") )
export_ouput_adata(test_ad, test_filen.name, out_data_path) # will append "_out.h5ad"


# ------------------------------------------
# In[19]:  TODO: save below to a separate script
# --------------
# ## LBL8R on scVI normalized expression  
# To give a "baseline" a fair shake its important to use normalized counts.  Using the `scVI` 
# normalization is our best shot... (Although the current models are NOT batch correcting 
# since we don't have a good strategy to do this with probe data)
out_data_path = data_path / ("LBL8R"+EMB)

# In[ ]: RE-LOAD TRAIN DATA
train_ad = ad.read_h5ad(train_filen)

# # this takes too much memory.  we need to create them from SCVI_nobatch and just load them

# #######################
# vae_model_path = model_root_path/ "SCVI_nobatch"
# vae_model_name = "scvi_nobatch"
# vae, train_ad = get_lbl8r_scvi( # sa,e ast get_trained_scvi but forces "batch"=None
#     train_ad,
#     labels_key=cell_type_key,
#     model_path=model_path,
#     model_name=vae_model_name,
# )

# train_ad = make_scvi_normalized_adata(vae, train_ad)


# in_path = out_data_path
# train_filen = in_path / XYLENA_TRAIN.replace("_cnt.h5ad", "_scvi_nb_out.h5ad")
# test_filen = in_path / XYLENA_TEST.replace("_cnt.h5ad", "_scvi_nb_out.h5ad")
# train_ad = ad.read_h5ad(train_filen)

# In[9]:
model_dir = "E2E_LBL8R"  #same as raw counts
cell_type_key = CELL_TYPE_KEY
out_path = data_path 

# In[6]:
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


retrain = True
plot_training = True
# In[ ]:

lbl8r_model_name = "scvi_expr"

labelator, train_ad = get_lbl8r( 
    train_ad,
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
    train_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=lbl8r_model_name,
    title_str="TRAIN",
    **fig_kwargs,
    )

# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(train_ad,
               basis=MDE_KEY,
                color=[cell_type_key, "batch"],
                device=device,
                **fig_kwargs,
                )

# save versions of test/train with latents and embeddings added
# 
# In[17]:
export_ouput_adata(train_ad, train_filen.name.replace(RAW, EXPR+NOBATCH), out_data_path)


# In[ ]:
# ------------------
# ## TEST
# ## 4.  Load data
test_ad = ad.read_h5ad(test_filen)
test_ad = make_scvi_normalized_adata(vae, test_ad)

# test_ad.obs["ground_truth"] = test_ad.obs[cell_type_key]

# In[19]:

exp_test_ad = query_lbl8r(
    test_ad,
    labelator,
    labels_key=cell_type_key,
)


# LBL8R.setup_anndata(test_ad, labels_key=cell_type_key)

# test_predictions = labelator.predict(test_ad, probs=False, soft=True)
# # In[20]:
# test_ad = add_predictions_to_adata(
#     test_ad, test_predictions, insert_key="pred", pred_key="label"
# )
# In[ ]:
plot_predictions(
    test_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=lbl8r_model_name,
    title_str="TEST",
    **fig_kwargs,

)

# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(test_ad,
               basis=MDE_KEY,
                color=[cell_type_key, "batch"],
                device=device,
                **fig_kwargs,
            )
# save versions of test/train with latents and embeddings added
# 
# In[23]:
export_ouput_adata(test_ad, test_filen.name.replace(RAW, EXPR+NOBATCH), out_data_path)

# %%
