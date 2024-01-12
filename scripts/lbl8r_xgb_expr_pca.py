#!/usr/bin/env python
# coding: utf-8
# 
# lbl8r xgb variants:
# - raw counts PCA loadings n=50 features
# - normalized counts (scVI) PCA loadings
# - scVI latent
# - etc.
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
            export_ouput_adata,
            )
from lbl8r import (
                get_xgb,
                query_xgb,
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
out_data_path = data_path / f"LBL8R{XGBOOST}"
# ## xgb_LBL8R on raw counts
# This is a zeroth order "baseline" for performance.
# ### load data
# ## 0. Load training data
in_data_path = data_path / f"LBL8R{PCS}"

train_filen = in_data_path / XYLENA_TRAIN.replace(H5,PCS+OUT+H5)
test_filen = in_data_path / XYLENA_TEST.replace(H5,PCS+OUT+H5)

# In[ ]:
model_dir = "XGB"
cell_type_key = CELL_TYPE_KEY
out_path = data_path / model_dir

# ## xgb_LBL8R on raw counts
# In[ ]:
# ## xgb_LBL8R on raw count PCAs 
# This is a zeroth order "baseline" for performance.
# 

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
    save = save,
    show = show, 
    fig_dir = fig_dir,
)

retrain=True

# In[ ]:
# load data and get label encoder
train_ad = ad.read_h5ad(train_filen)
xgb_model_name = "xgb_raw_pcs"
# In[ ]:

bst,train_ad, le = get_xgb(
    train_ad,
    label_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=xgb_model_name,
)

# In[ ]:
# 1. add the predictions to the adata
plot_predictions(train_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=xgb_model_name, 
                 title_str="TRAIN",
                 **fig_kwargs,

            )
# In[ ]:
# ### test
test_ad = ad.read_h5ad(test_filen)

test_ad, test_report = query_xgb(test_ad, bst, le)
plot_predictions(test_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=xgb_model_name, 
                 title_str="TEST",
                 **fig_kwargs,
            )

# 
# In[23]:
# ## 7: save versions of test/train with latents and embeddings added
export_ouput_adata(train_ad, train_filen.name, out_data_path)
export_ouput_adata(test_ad, test_filen.name, out_data_path)

# ------------------------------------------------
# In[ ]:
# ## XGB LBL8R on scVI normalized PCAs 
# To give the pca "baseline" a fair shake its important to use normalized counts. 
#  Using the `scVI` normalization is our best shot... (Although the current models 
#  are NOT batch correcting since we don't have a good strategy to do this with probe data)
in_data_path = data_path / f"LBL8R{EXPR}{PCS}"

train_filen = in_data_path / XYLENA_TRAIN.replace(RAW, EXPR+PCS+OUT)
test_filen = in_data_path / XYLENA_TEST.replace(RAW, EXPR+PCS+OUT)

# In[ ]:
# load data and get label encoder
train_ad = ad.read_h5ad(train_filen)
xgb_model_name = "xgb_expr_pcs"
# In[ ]:

bst,train_ad, le = get_xgb(
    train_ad,
    label_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=xgb_model_name,
)

# In[ ]:
# 1. add the predictions to the adata
plot_predictions(train_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=xgb_model_name, 
                 title_str="TRAIN",
                  **fig_kwargs,
            )
# In[ ]:
# ### test
test_ad = ad.read_h5ad(test_filen)

# In[ ]:
test_ad, test_report = query_xgb(test_ad, bst, le)

# In[ ]:

plot_predictions(test_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=xgb_model_name, 
                 title_str="TEST",
                  **fig_kwargs,
            )

# 
# In[23]:
# ## 7: save versions of test/train with latents and embeddings added
export_ouput_adata(train_ad, train_filen.name, out_data_path)
export_ouput_adata(test_ad, test_filen.name, out_data_path)



# ------------------------------------------------
# In[ ]:
# ## xgb_LBL8R on scVI latents  
# 
in_data_path = data_path / "LBL8R_scvi"

train_filen = in_data_path / XYLENA_TRAIN.replace(RAW, EMB+OUT)
test_filen = in_data_path / XYLENA_TEST.replace(RAW, EMB+OUT)

# In[ ]:
# load data and get label encoder

# ## 0. Load training data

train_ad = ad.read_h5ad(train_filen)
xgb_model_name = "xgb_scvi_nb"
# In[ ]:

bst,train_ad, le = get_xgb(
    train_ad,
    label_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=xgb_model_name,
)

# In[ ]:
# 1. add the predictions to the adata
plot_predictions(train_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=xgb_model_name, 
                 title_str="TRAIN",
                  **fig_kwargs,

            )
# In[ ]:
# ### test
test_ad = ad.read_h5ad(test_filen)

test_ad, test_report = query_xgb(test_ad, bst, le)
plot_predictions(test_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=xgb_model_name, 
                 title_str="TEST",
                  **fig_kwargs,

            )

# 
# In[23]:
# ## 7: save versions of test/train with latents and embeddings added
export_ouput_adata(train_ad, train_filen.name, out_data_path)
export_ouput_adata(test_ad, test_filen.name, out_data_path)



# %%
