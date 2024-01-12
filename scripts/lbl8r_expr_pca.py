#!/usr/bin/env python
# coding: utf-8

# ## imports / parameters
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
                get_pca_lbl8r, 
                query_lbl8r, 
                prep_lbl8r_adata,
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
# --------------
# ## pcaLBL8R on PCAs of scVI normalized expression 
# To give the pca "baseline" a fair shake its important to use normalized counts.  
# Using the `scVI` normalization is our best shot... (Although the current models 
# are NOT batch correcting since we don't have a good strategy to do this with probe data)
# In[ ]:
out_data_path = data_path / "LBL8R"+EXPR+PCS

# ## 0. Load training data
# get the input data from lbl8r_scvi.py output
in_path = data_path / "LBL8R"+EMB

train_filen = in_path / XYLENA_TRAIN.replace(RAW, EXPR+NOBATCH)
test_filen = in_path / XYLENA_TEST.replace(RAW, EXPR+NOBATCH)

train_ad = ad.read_h5ad(train_filen)

# In[ ]:
model_dir = "EXPR_pca"
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

if fig_dir is not None:
    fig_dir = Path(fig_dir) / model_dir
    if not fig_dir.exists():
        fig_dir.mkdir()
    

retrain = True
plot_training = True

# In[ ]:
pca_model_name = "lbl8r_expr_pca"
pca_train_ad = prep_lbl8r_adata(train_ad, pca_key=PCA_KEY, labels_key=cell_type_key)

labelator, train_ad = get_pca_lbl8r( #get_lbl8r
    pca_train_ad,
    labels_key=cell_type_key,
    model_path=model_path,
    retrain=retrain,
    model_name=pca_model_name,
    plot_training=plot_training,
    **fig_kwargs,
)

# In[ ]:
# ## 3: visualize prediction fidelity on training set
plot_predictions(
    pca_train_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=pca_model_name,
    title_str="TRAIN",
    **fig_kwargs,
    )

# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(pca_train_ad,
               basis=MDE_KEY,
                color=[cell_type_key, "batch"],
                device=device,
               **fig_kwargs,
                )

# ---------------------------------------------------
# In[ ]:
# ------------------
# ## TEST
# ## 4.  Load data
test_ad = ad.read_h5ad(test_filen)


# In[ ]:
# ## 5 - prep lbl8r adata and query (run) model
pca_test_ad = prep_lbl8r_adata(test_ad, pca_key=PCA_KEY, labels_key=cell_type_key)
pca_test_ad = query_lbl8r(
    pca_test_ad,
    labelator,
    labels_key=cell_type_key,
)

# In[ ]:
plot_predictions(
    pca_test_ad,
    pred_key="pred",
    cell_type_key=cell_type_key,
    model_name=pca_model_name,
    title_str="TEST",
    **fig_kwargs,
)

# In[ ]:
# this should also add the embeddings to the adata
plot_embedding(pca_test_ad,
               basis=MDE_KEY,
                color=[cell_type_key, "batch"],
                device=device,
                **fig_kwargs,
            )

# In[ ]:
export_ouput_adata(pca_train_ad, train_filen.name.replace(H5,PCS+H5), out_data_path)
export_ouput_adata(pca_test_ad, test_filen.name.replace(H5,PCS+H5), out_data_path)



