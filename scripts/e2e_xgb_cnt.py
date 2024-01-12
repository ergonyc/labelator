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
            plot_embedding,
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
# setup 1
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
# setup 2: XGBoost on raw counts
# This is a zeroth order "baseline" for performance.
out_data_path = data_path / "XGB"

train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST

# In[ ]:
# setup 3
model_dir = "XGB"
cell_type_key = CELL_TYPE_KEY

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
# plot_training = True no training plots for XGB

# In[ ]:
# load data and get label encoder
train_ad = ad.read_h5ad(train_filen)
xgb_model_name = "xgb_raw_cnt"

# In[ ]:
# Get the trained xgb model
bst,train_ad, le = get_xgb(
    train_ad,
    label_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=xgb_model_name,
)

# In[ ]:
# training prediction results
plot_predictions(train_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=xgb_model_name, 
                 title_str="TRAIN",
                 **fig_kwargs,
            )

# In[ ]:
# visualize embedding 
plot_embedding(
    train_ad,
    basis=MDE_KEY,
    color=["pred", "batch"],
    device=device,
    **fig_kwargs,
)


# In[ ]:
# ### test the model
test_ad = ad.read_h5ad(test_filen)

test_ad, test_report = query_xgb(test_ad, bst, le)

# In[ ]:
# testingng prediction results
plot_predictions(test_ad, 
                 pred_key="pred", 
                 cell_type_key=cell_type_key, 
                 model_name=xgb_model_name, 
                 title_str="TEST",
                  **fig_kwargs,
            )

# In[ ]:
# visualize
plot_embedding(
    test_ad,
    basis=MDE_KEY,
    color=["pred", "batch"],
    device=device,
    **fig_kwargs,
)

# 
# In[ ]:
# ## save versions of test/train with latents and embeddings added
export_ouput_adata(train_ad, train_filen.name, out_data_path)
export_ouput_adata(test_ad, test_filen.name, out_data_path)

# ------------------
# TODO:  evaluation for entropy of predictions
# 
# 
# TODO:  strategy for "Unknown" low-quality predictions
