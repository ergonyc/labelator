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

#
# TODO:  add options for model configureaion: e.g. n_hidden, n_latent, n_layers, dropout_rate, dispersion, gene_likelihood, latent_distribution encode_covariates,
# TODO: enable other **training_kwargs:  train_size, accelerators, devices, early_stopping, early_stopping_kwargs, batch_size, epochs, etc.


# TODO: add logging
train_path = Path("data/scdata/xylena/brain_atlas_anndata_train_cnt.h5ad")
query_path = Path("data/scdata/xylena/brain_atlas_anndata_test_cnt.h5ad")

# train_path = None
# query_path = Path('data/scdata/ASAP/artifacts/06_merged_filtered_processed_integrated_clustered_anndata_object.h5ad')
# # query_path = Path('data/scdata/ASAP/artifacts/06_merged_filtered_integrated_clustered_anndata_object.h5ad')
# query_path = Path('data/scdata/ASAP/artifacts/07_merged_filtered_integrated_clustered_annotated_anndata_object.h5ad')
# model_path = Path("models/CNT/")
model_path = Path("models/REPR/scvi/")
# model_path = Path("models/TRANSFER/")
# train_path = None
# model_name = "raw_lbl8r"
model_name = "scvi_emb_xgb"
# model_name = "pcs_lbl8r"
# model_name = "scvi_emb"
# model_name = "scanvi_batch_eq"

output_data_path = Path("data/scdata/xylena/LABELATOR/")
artifacts_path = Path("artifacts/")
gen_plots = True
retrain_model = False
labels_key = CELL_TYPE_KEY

# if model_name == "scanvi_batch_eq":
#     batch_key = "sample"
# else:
#     batch_key = None

%load_ext autoreload
%autoreload 2

# In[ ]
""" Command line interface for model processing pipeline.
"""

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

# dummy_label = train_data.adata.obs[CELL_TYPE_KEY].values[0]
# ad = query_data.adata

# ad.obs[CELL_TYPE_KEY] = dummy_label
# # update data with ad
# query_data.update(ad)
# In[ ]/'

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
