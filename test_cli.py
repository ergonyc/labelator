# import click
# In[ ]
import torch
from pathlib import Path

from lbl8r.labelator import (
    load_data,
    prep_model,
    query_model,
    prep_query_model,
    archive_artifacts,
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
query_path = Path("data/scdata/xylena/brain_atlas_anndata_test_cnt.h5ad")
train_path = Path("data/scdata/xylena5k/xyl2_train.h5ad")
query_path = Path("data/scdata/xylena5k/xyl2_test.h5ad")
# query_path = Path("data/scdata/xylena5k/xyl2_query.h5ad")
# train_path = None

# query_path = Path('data/scdata/ASAP/artifacts/06_merged_filtered_processed_integrated_clustered_anndata_object.h5ad')
# # query_path = Path('data/scdata/ASAP/artifacts/06_merged_filtered_integrated_clustered_anndata_object.h5ad')
# query_path = Path('data/scdata/ASAP/artifacts/07_merged_filtered_integrated_clustered_annotated_anndata_object.h5ad')
model_path = Path("models5k/REPR/scvi/")
# model_name = "scvi_emb_xgb"
model_name = "scvi_emb"
# model_name = "scvi_expr"
# model_name = "scvi_expr_pcs"

# model_path = Path("models5k/TRANSFER/")
# model_name = "scanvi_batch_eq"

# model_path = Path("models5k/CNT/")
# model_name = "pcs_lbl8r"
# model_name = "raw_lbl8r"

output_data_path = Path("data/scdata/xylena5k/LABELATOR/")
artifacts_path = Path("artifacts5k/")


train_path = Path("data/scdata/xylena10k/xyl2_train.h5ad")
query_path = Path("data/scdata/xylena10k/xyl2_test.h5ad")
model_path = Path("models10k/REPR/scvi/")
model_name = "scvi_emb"

output_data_path = Path("data/scdata/xylena10k/LABELATOR/")
artifacts_path = Path("artifacts10k/")



gen_plots = True
retrain_model = False
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
if train := train_path is not None:
    train_data = load_data(train_path, archive_path=output_data_path)
else:
    if retrain_model:
        print("Must provide training data (`train-path`) to retrain model")
    train_data = None

if query := query_path is not None:
    query_data = load_data(query_path, archive_path=output_data_path)
    # load model with query_data if training data is not provided
else:
    query_data = None

if not (train | query):
    print("Must provide either `data-path` or `query-path` or both")


# In[ ]
# # hack to keep only marker genes
# ad = query_data.adata
# ad = ad[:, keep_genes].copy()
# query_data.update(ad)

# In[ ]
## PREP MODEL ###################################################################
# gets model and preps Adata
# TODO:  add additional training_kwargs to cli
training_kwargs = {} #dict(batch_key=batch_key)
print(f"prep_model: {'🛠️ '*25}")

model_set, train_data = prep_model(
    train_data,  # Note this is actually query_data if train_data arg was None
    model_name=model_name,
    model_path=model_path,
    labels_key=labels_key,
    retrain=retrain_model,
    **training_kwargs,
)
# In[ ]
## QUERY MODELs ###################################################################
# makes sure the genes correspond to those of the prepped model
#     projects counts onto the principle components of the training datas eigenvectors as 'X_pca'
# TODO:  add additional training_kwargs to cli
if query:
    print(f"prep query: {'💅 '*25}")
    model_set, query_data = prep_query_model(
        query_data,
        model_set,
        model_name,
        labels_key=labels_key,
        retrain=retrain_model,
    )


# In[ ]
if train:
    # prep_train_data
    #    - check if train_data was prepped (i.e. model was trained in prep_model)
    #    - if not, prep_train_data
    print(f"train_model: {'🏋️ '*25}")
    train_data = query_model(train_data, model_set)

if query:
    print(f"query_model: {'🔮 '*25}")
    query_data = query_model(query_data, model_set)
# In[ ]
## CREATE ARTIFACTS ###################################################################

if gen_plots:
    archive_artifacts(
        train_data,
        query_data,
        model_set,
        path=artifacts_path,
    )


# %%
