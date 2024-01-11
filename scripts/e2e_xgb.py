#!/usr/bin/env python
# coding: utf-8
# 
# ec e2e xgb variants:
# - raw counts 
# - normalized counts (scVI) (no batch)
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
out_data_path = data_path / "XGB"
# ## xgb_LBL8R on raw counts
# This is a zeroth order "baseline" for performance.
# ### load data
train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST


# In[ ]:
model_dir = "XGB"
cell_type_key = "cell_type"
out_path = data_path / model_dir

model_root_path = root_path / "lbl8r_models"
if not model_root_path.exists():
    model_root_path.mkdir()

model_path = model_root_path / model_dir
if not model_path.exists():
    model_path.mkdir()

retrain=True

# In[ ]:
# load data and get label encoder
train_ad = ad.read_h5ad(train_filen)
xgb_model_name = "xgb_raw_cnt"
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
                save = save,
                show = show, 
                fig_dir = fig_dir,
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
                save = save,
                show = show, 
                fig_dir = fig_dir,
            )

# 
# In[23]:
# test_ad.write_h5ad(out_data_path / test_filen.name.replace(".h5ad", "_xgb.h5ad") )
# train_ad.write_h5ad(out_data_path / test_filen.name.replace(".h5ad", "_xgb.h5ad") )


# ## 7: save versions of test/train with latents and embeddings added
export_ouput_adata(train_ad, train_filen.name.replace(".h5ad", "_xgb.h5ad"), out_data_path)
export_ouput_adata(test_ad, test_filen.name.replace(".h5ad", "_xgb.h5ad"), out_data_path)

# --------------
# 
# ## scVI normalized counts

# 
# ### load data

# In[9]:

data_path = data_path / "LBL8R"

train_filen = data_path / XYLENA_TRAIN.replace("_cnt.h5ad", "_exp_nb_out.h5ad")
test_filen = data_path / XYLENA_TEST.replace("_cnt.h5ad", "_exp_nb_out.h5ad")

xgb_model_name = "xgb_exp_nb"

# In[ ]:
# load data and get label encoder
train_ad = ad.read_h5ad(train_filen)

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
                save = save,
                show = show, 
                fig_dir = fig_dir,
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
                save = save,
                show = show, 
                fig_dir = fig_dir,
            )

# 
# In[23]:
# test_ad.write_h5ad(out_data_path / test_filen.name.replace(".h5ad", "_xgb.h5ad") )
# train_ad.write_h5ad(out_data_path / test_filen.name.replace(".h5ad", "_xgb.h5ad") )


# ## 7: save versions of test/train with latents and embeddings added
export_ouput_adata(train_ad, train_filen.name.replace(".h5ad", "_xgb.h5ad"), out_data_path)
export_ouput_adata(test_ad, test_filen.name.replace(".h5ad", "_xgb.h5ad"), out_data_path)








# ------------------
# TODO:  evaluation for entropy of predictions
# 
# 
# TODO:  strategy for "Unknown" low-quality predictions
