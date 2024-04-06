# import click
# In[ ]
import torch
from pathlib import Path

from lbl8r.labelator import (
    load_data,
    prep_model,
    query_model,
    get_trained_model,
    load_trained_model,
    prep_query_model,
    archive_plots,
    archive_data,
    CELL_TYPE_KEY,
)
# In[ ]

# #
# repr_model_names=["scvi_emb", "scvi_expr", "scvi_expr_pcs"]
# count_model_names=["pcs_lbl8r", "raw_lbl8r"]
# transfer_model_names=["scanvi_batch_eq", "scanvi"]

# train_data="data/scdata/xylena5k/xyl2_train.h5ad"
# query_data="data/scdata/xylena5k/xyl2_test.h5ad"
# adata_output_path='data/scdata/xylena5k/LABELATOR/'
# artifacts_path='artifacts5k/'

# model_path='models5k/REPR/scvi'  


train_path = Path("data/scdata/xylena/brain_atlas_anndata_train_cnt.h5ad")
SET_NAME = '3k'
train_path = Path(f"data/scdata/xylena{SET_NAME}/xyl2_train.h5ad")


# query_path = Path('data/scdata/ASAP/artifacts/06_merged_filtered_processed_integrated_clustered_anndata_object.h5ad')
# # query_path = Path('data/scdata/ASAP/artifacts/06_merged_filtered_integrated_clustered_anndata_object.h5ad')
# query_path = Path('data/scdata/ASAP/artifacts/07_merged_filtered_integrated_clustered_annotated_anndata_object.h5ad')

model_path = Path(f"models{SET_NAME}/REPR/scvi/")
# model_name = "scvi_emb_xgb"
model_name = "scvi_emb"
# model_name = "scvi_expr"
# model_name = "scvi_expr_pcs"
# model_path = Path("models5k/TRANSFER/")
# model_name = "scanvi_batch_eq"
# model_path = Path("models5k/CNT/")
# model_name = "pcs_lbl8r"
# model_name = "raw_lbl8r"
output_data_path = Path(f"data/scdata/xylena{SET_NAME}/LABELATOR/")
artifacts_path = Path(f"artifacts{SET_NAME}/")


SET_NAME = '3k'
train_path = Path(f"data/scdata/xylena{SET_NAME}/xyl2_train.h5ad")
query_path = Path(f"data/scdata/xylena{SET_NAME}/xyl2_test.h5ad")
model_path = Path(f"models{SET_NAME}/REPR/scvi/")
model_name = "scvi_emb"

# output_data_path = Path("data/scdata/xylena15k/LABELATOR/")
# artifacts_path = Path("artifacts15k/")

gen_plots = True
retrain_model = True
labels_key = CELL_TYPE_KEY
# labels_key = "cell_type"
# if model_name == "scanvi_batch_eq":
#     batch_key = "sample"
# else:
#     batch_key = None

%load_ext autoreload
%autoreload 2

# In[ ]
""" Command line interface for model processing pipeline.
"""
# import pandas as pd
# XYLENA2_GROUND_TRUTH = "ground_truth_labels.csv"
# XYLENA2_RAW_PATH = "data/scdata/xylena_raw"

# root_path = Path.cwd()
# raw_data_path = root_path / XYLENA2_RAW_PATH

# gene_list = pd.read_csv(raw_data_path / "xyl2_full_hvg.csv", index_col=0)

# n_top_gene = 500
# keep_genes = gene_list.iloc[:n_top_gene].index.to_list()

# setup
torch.set_float32_matmul_precision("medium")

# print(
#         f"{train_path=}:: {query_path=}:: {model_path=}:: {model_name=}:: {output_data_path=}:: {artifacts_path=}:: {gen_plots=}:: {retrain_model=}:: {labels_key=}:: {batch_key=}"
#     )
## LOAD DATA ###################################################################
train_data = load_data(train_path, archive_path=output_data_path)

# gets model and preps Adata
# TODO:  add additional training_kwargs to cli
training_kwargs = {}  # dict(batch_key=batch_key)
print(f"prep_model: {'🛠️ '*25}")

train_data.labels_key = labels_key
model, train_data = get_trained_model(
    train_data,
    model_name,
    model_path,
    labels_key=labels_key,
    retrain=retrain_model,
    **training_kwargs,
)

# WARNING:  BUG.  if train_data is None preping with query data hack won't work for PCs
model_set, train_data = prep_model(
    train_data,  # Note this is actually query_data if train_data arg was None
    model_name=model_name,
    model_path=model_path,
    labels_key=labels_key,
    retrain=retrain_model,
    **training_kwargs,
)

# In[ ]
# prep_train_data
#    - check if train_data was prepped (i.e. model was trained in prep_model)
#    - if not, prep_train_data
print(f"train_model: {'🏋️ '*25}")
train_data = query_model(train_data, model_set)

# In[ ]
## CREATE ARTIFACTS ###################################################################

if gen_plots:
    print(f"archive training plots and data: {'📈 '*25}")
    archive_plots(
        train_data, model_set, "train", fig_path=(artifacts_path / "figs")
    )
    print(f"archive train output adata: {'💾 '*25}")
    archive_data(train_data)
# %%




###   QUERY  query_cli.py ####################################################################



query_path = Path("data/scdata/xylena/brain_atlas_anndata_test_cnt.h5ad")
query_path = Path("data/scdata/xylena5k/xyl2_test.h5ad")

query_path = Path("data/scdata/xylena3k/xyl2_query.h5ad")

# In[ ]
## QUERY MODELs ###################################################################
# makes sure the genes correspond to those of the prepped model
#     projects counts onto the principle components of the training datas eigenvectors as 'X_pca'
# TODO:  add additional training_kwargs to cli

## LOAD DATA ###################################################################
query_data = load_data(query_path, archive_path=output_data_path)

## PREP MODEL ###################################################################
# gets model and preps Adata
# TODO:  add additional training_kwargs to cli
training_kwargs = {}  # dict(batch_key=batch_key)
print(f"prep_model: {'🛠️ '*25}")

model_set = load_trained_model(model_name, model_path, labels_key=labels_key)
# if no traing data is loaded (just prepping for query) return placeholder data

# In[ ]
## QUERY MODELs ###################################################################
# makes sure the genes correspond to those of the prepped model
#     projects counts onto the principle components of the training datas eigenvectors as 'X_pca'
# TODO:  add additional training_kwargs to cli
print(f"prep query: {'💅 '*25}")
# prep query model actually preps data unless its a scANVI model...
#
model_set, query_data = prep_query_model(
    query_data,
    model_set,
    model_name,
    labels_key=labels_key,
    retrain=retrain_model,
)

# In[ ]
print(f"query_model: {'🔮 '*25}")
query_data = query_model(query_data, model_set)
# In[ ]
## CREATE ARTIFACTS ###################################################################
if gen_plots:
    print(f"archive query plots and data: {'📊 '*25}")
    archive_plots(
        query_data, model_set, "query", fig_path=(artifacts_path / "figs")
    )

print(f"archive query plots and data: {'📊 '*25}")
archive_plots(query_data, model_set, "query", fig_path=(artifacts_path / "figs"))
print(f"archive query output adata: {'💾 '*25}")
archive_data(query_data)
