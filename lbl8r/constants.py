

import sys
import os
from pathlib import Path

## adata keys
SCVI_LATENT_KEY = "X_scVI"
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTIONS_KEY = "C_scANVI"
PCA_KEY = "X_pca"


## export name parts
EXPR = "_EXPR"
OUT = "out"
H5 = ".h5ad"
PCS = "_pcs"
NOBATCH = "_nb"
EMB = "_scvi"
MDE = "_mde"

# exports:  where the model is saved
MODEL_SAVE_DIR = "lbl8r_models"

## xylena data names
XYLENA_ANNDATA = "brain_atlas_anndata.h5ad"
XYLENA_TRAIN = XYLENA_ANNDATA.replace(".h5ad", "_train_cnt.h5ad")
XYLENA_TEST = XYLENA_ANNDATA.replace(".h5ad", "_test_cnt.h5ad")
XYLENA_TRAIN_SPARSE = XYLENA_TRAIN.replace(".h5ad", "_sparse.h5ad")
XYLENA_TEST_SPARSE = XYLENA_TEST.replace(".h5ad", "_sparse.h5ad")

## path to data (to be overridden by CLI or user scripts)
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    XYLENA_PATH = "/content/drive/MyDrive/SingleCellModel/data"
else:
    if sys.platform == "darwin":
        XYLENA_PATH = "data/xylena"
        XYLENA_RAW_PATH = "data/xylena_raw"
    else: #linux
        XYLENA_PATH = "data/scdata/xylena"
        XYLENA_RAW_PATH = "data/scdata/xylena_raw"
        

__all__ = [
    "SCVI_LATENT_KEY",
    "SCANVI_LATENT_KEY",
    "SCANVI_PREDICTIONS_KEY",
    "PCA_KEY",
    "EXPR",
    "OUT",
    "H5",
    "PCS",
    "NOBATCH",
    "EMB",
    "MDE",
    "MODEL_SAVE_DIR",
    "XYLENA_ANNDATA",
    "XYLENA_TRAIN",
    "XYLENA_TEST",
    "XYLENA_TRAIN_SPARSE",
    "XYLENA_TEST_SPARSE",
]
