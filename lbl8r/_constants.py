import sys
import os
from pathlib import Path

## adata keys
SCVI_LATENT_KEY = "X_scVI"
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTIONS_KEY = "C_scANVI"
PCA_KEY = "X_pca"
SCVI_MDE_KEY = "X_scVI_mde"
MDE_KEY = "X_mde"

CELL_TYPE_KEY = "cell_type"

## export name parts
EXPR = "_expr"
OUT = "_out"
H5 = ".h5ad"
PCS = "_pcs"
XGBOOST = "_xgb"
NOBATCH = "_nb"
EMB = "_emb"
MDE = "_mde"
RAW = "_cnt"
TRAIN = "_train"
TEST = "_test"
SPARSE = "_sparse"

# exports:  where the model is saved
MODEL_SAVE_DIR = "lbl8r_models"

## xylena data names
XYLENA_ANNDATA = "brain_atlas_anndata.h5ad"
XYLENA_TRAIN = XYLENA_ANNDATA.replace(H5, TRAIN + RAW + H5)
XYLENA_TEST = XYLENA_ANNDATA.replace(H5, TEST + RAW + H5)

XYLENA_TRAIN_SPARSE = XYLENA_TRAIN.replace(H5, SPARSE + H5)
XYLENA_TEST_SPARSE = XYLENA_TEST.replace(H5, SPARSE + H5)

## path to data (to be overridden by CLI or user scripts)
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    XYLENA_PATH = "/content/drive/MyDrive/SingleCellModel/data"
else:
    if sys.platform == "darwin":
        XYLENA_PATH = "data/xylena"
        XYLENA_RAW_PATH = "data/xylena_raw"
    else:  # linux
        XYLENA_PATH = "data/scdata/xylena"
        XYLENA_RAW_PATH = "data/scdata/xylena_raw"

XYLENA_METADATA = "final_metadata.csv"
XYLENA_ANNDATA2 = "brain_atlas_anndata_updated.h5ad"

__all__ = [
    "SCVI_LATENT_KEY",
    "SCANVI_LATENT_KEY",
    "SCANVI_PREDICTIONS_KEY",
    "PCA_KEY",
    "SCVI_MDE_KEY",
    "MDE_KEY",
    "CELL_TYPE_KEY",
    "EXPR",
    "OUT",
    "H5",
    "PCS",
    "XGBOOST",
    "NOBATCH",
    "EMB",
    "MDE",
    "RAW",
    "TRAIN",
    "TEST",
    "SPARSE",
    "MODEL_SAVE_DIR",
    "XYLENA_ANNDATA",
    "XYLENA_TRAIN",
    "XYLENA_TEST",
    "XYLENA_TRAIN_SPARSE",
    "XYLENA_TEST_SPARSE",
]
