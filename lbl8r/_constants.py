import sys
import os
from pathlib import Path

## adata keys
SCVI_LATENT_KEY = "X_scVI"

SCVI_LATENT_KEY_Z = SCVI_LATENT_KEY
SCVI_LATENT_KEY_MU_VAR = "X_scVI_mu_var"

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
CNT = "_cnt"
TRAIN = "_train"
TEST = "_test"
SPARSE = "_sparse"

# SCANVI/SCVI model names
SCVI_SUB_MODEL_NAME = "scvi"
SCANVI_SUB_MODEL_NAME = "scanvi"
QUERY_SCVI_SUB_MODEL_NAME = "query_scvi"
QUERY_SCANVI_SUB_MODEL_NAME = "query_scanvi"
LBL8R_SCVI_SUB_MODEL_NAME = "scvi_emb"

# LBL8R model names
SCVI_LATENT_MODEL_NAME = "lbl8r_scvi_emb"
RAW_PC_MODEL_NAME = "lbl8r_raw_cnt_pcs"
SCVI_EXPR_PC_MODEL_NAME = "lbl8r_scvi_expr_pcs"

# LBL8R XGBoost model names
XGB_SCVI_LATENT_MODEL_NAME = "xgb_scvi_emb"
XGB_RAW_PC_MODEL_NAME = "xgb_raw_cnt_pcs"
XGB_SCVI_EXPR_PC_MODEL_NAME = "xgb_scvi_expr_pcs"

# E2E model names
# lbl8r
LBL8R_SCVI_EXPRESION_MODEL_NAME = "lbl8r_scvi_expr"
LBL8R_RAW_COUNT_MODEL_NAME = "lbl8r_raw_cnt"
# scanvi
SCANVI_BATCH_EQUALIZED_MODEL_NAME = "scanvi_batch_equal"
SCANVI_MODEL_NAME = "scanvi"
# e2e XGBoost model names
XGB_SCVI_EXPRESION_MODEL_NAME = "xgb_scvi_expr"
XGB_RAW_COUNT_MODEL_NAME = "xgb_raw_cnt"

VALID_MODEL_NAMES = [
    SCVI_LATENT_MODEL_NAME,
    RAW_PC_MODEL_NAME,
    SCVI_EXPR_PC_MODEL_NAME,
    XGB_SCVI_LATENT_MODEL_NAME,
    XGB_RAW_PC_MODEL_NAME,
    XGB_SCVI_EXPR_PC_MODEL_NAME,
    LBL8R_SCVI_EXPRESION_MODEL_NAME,
    LBL8R_RAW_COUNT_MODEL_NAME,
    SCANVI_BATCH_EQUALIZED_MODEL_NAME,
    SCANVI_MODEL_NAME,
    XGB_SCVI_EXPRESION_MODEL_NAME,
    XGB_RAW_COUNT_MODEL_NAME,
]


__all__ = [
    "SCVI_LATENT_KEY",
    "SCVI_LATENT_KEY_Z",
    "SCVI_LATENT_KEY_MU_VAR",
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
    "CNT",
    "TRAIN",
    "TEST",
    "SPARSE",
    "SCVI_SUB_MODEL_NAME",
    "SCANVI_SUB_MODEL_NAME",
    "QUERY_SCVI_SUB_MODEL_NAME",
    "QUERY_SCANVI_SUB_MODEL_NAME",
    "LBL8R_SCVI_SUB_MODEL_NAME",
    "SCVI_LATENT_MODEL_NAME",
    "RAW_PC_MODEL_NAME",
    "SCVI_EXPR_PC_MODEL_NAME",
    "XGB_SCVI_LATENT_MODEL_NAME",
    "XGB_RAW_PC_MODEL_NAME",
    "XGB_SCVI_EXPR_PC_MODEL_NAME",
    "LBL8R_SCVI_EXPRESION_MODEL_NAME",
    "LBL8R_RAW_COUNT_MODEL_NAME",
    "SCANVI_BATCH_EQUALIZED_MODEL_NAME",
    "SCANVI_MODEL_NAME",
    "XGB_SCVI_EXPRESION_MODEL_NAME",
    "XGB_RAW_COUNT_MODEL_NAME",
    "VALID_MODEL_NAMES",
]


# exports:  where the model is saved
MODEL_SAVE_DIR = "lbl8r_models"

## xylena data names
XYLENA_ANNDATA = "brain_atlas_anndata.h5ad"
XYLENA_TRAIN = XYLENA_ANNDATA.replace(H5, TRAIN + CNT + H5)
XYLENA_TEST = XYLENA_ANNDATA.replace(H5, TEST + CNT + H5)

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
