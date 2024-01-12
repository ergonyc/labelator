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
sys.path.append(os.path.abspath((os.path.join(os.getcwd(), ".."))))

from lbl8r.utils import (
    plot_predictions,
    plot_embedding,
    export_ouput_adata,
)
from lbl8r import (
    get_lbl8r,
    query_lbl8r,
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


# In[8]:
# Setup 2: lbl8r classification on raw count data
out_data_path = data_path / f"LBL8R{RAW}"

# ##  training data
train_filen = data_path / XYLENA_TRAIN
test_filen = data_path / XYLENA_TEST

# In[9]:
# setup 3
model_dir = "E2E_LBL8R"
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

fig_kwargs = dict(
    save=save,
    show=show,
    fig_dir=fig_dir,
)

retrain = True
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
plot_embedding(
    train_ad,
    basis=MDE_KEY,
    color=[cell_type_key, "batch"],
    device=device,
    **fig_kwargs,
)

# save versions of test/train with latents and embeddings added
#
# In[ ]:
# export training data w/updates
export_ouput_adata(train_ad, train_filen.name, out_data_path)

# In[ ]:
# ------------------
# ## TEST
# ## 4.  Load data
test_ad = ad.read_h5ad(test_filen)

# test_ad.obs["ground_truth"] = test_ad.obs[cell_type_key]

# In[ ]:
test_ad = query_lbl8r(
    test_ad,
    labelator,
    labels_key=cell_type_key,
)

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
plot_embedding(
    test_ad,
    basis=PCA_KEY,
    color=[cell_type_key, "batch"],
    device=device,
    **fig_kwargs,
)
# save versions of test/train with latents and embeddings added
#
# In[ ]:
# export training data w/updates
export_ouput_adata(test_ad, test_filen.name, out_data_path)  # will append "_out.h5ad"
