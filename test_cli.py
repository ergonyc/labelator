# import click
# In[ ]
import torch
from pathlib import Path

from lbl8r.labelator import (
    load_training_data,
    load_query_data,
    prep_model,
    query_model,
    prep_query_model,
    archive_plots,
    archive_data,
    CELL_TYPE_KEY,

)

#
# TODO:  add options for model configureaion: e.g. n_hidden, n_latent, n_layers, dropout_rate, dispersion, gene_likelihood, latent_distribution encode_covariates,
# TODO: enable other **training_kwargs:  train_size, accelerators, devices, early_stopping, early_stopping_kwargs, batch_size, epochs, etc.


# TODO: add logging
data_path = Path("data/scdata/xylena/brain_atlas_anndata_train_cnt.h5ad")
query_path = Path("data/scdata/xylena/brain_atlas_anndata_test_cnt.h5ad")
model_path = Path("models/CNT/")
# model_path = Path("models/REPR/scvi/")

model_name = "pcs_lbl8r"
model_name = "scvi_emb_xgb"
model_name = "pcs_xgb"
output_data_path = Path("data/scdata/xylena/LBL8R/")
artifacts_path = Path("artifacts/")
gen_plots = True
retrain_model = False
labels_key = CELL_TYPE_KEY

if model_name == "scanvi_batch_eq":
    batch_key = "sample"
else:
    batch_key = None

%load_ext autoreload
%autoreload 2

# In[ ]
"""
Command line interface for model processing pipeline.
"""

# setup
torch.set_float32_matmul_precision("medium")

## LOAD DATA ###################################################################
if train := data_path is not None:
    train_data = load_training_data(data_path)
else:
    if retrain_model:
        print("Must provide training data (`data-path`) to retrain model")
    train_data = None

if query := query_path is not None:
    query_data = load_query_data(query_path)
    # load model with query_data if training data is not provided
    if train_data is None:
        train_data = query_data

else:
    query_data = None

if not (train | query):
    print("Must provide either `data-path` or `query-path` or both")

# In[ ]/'

## PREP MODEL ###################################################################
# gets model and preps Adata
# TODO:  add additional training_kwargs to cli
training_kwargs = dict(batch_key=batch_key)
print(f"prep_model: {'🛠️ '*25}")

model, train_data = prep_model(
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
    query_data, model = prep_query_model(
        query_data,
        model,
        model_name,
        train_data,
        labels_key=labels_key,
        retrain=retrain_model,
    )


# In[ ]
if train:
    # prep_train_data  
    #    - check if train_data was prepped (i.e. model was trained in prep_model)
    #    - if not, prep_train_data
    print(f"train_model: {'🏋️ '*25}")
    train_data = query_model(train_data, model.model[model_name])

if query:
    print(f"query_model: {'🔮 '*25}")

    query_data = query_model(query_data, model.model[model_name])
# In[ ]
## CREATE ARTIFACTS ###################################################################
# TODO:  wrap in Models, Figures, and Adata in Artifacts class.
#       currently the models are saved as soon as they are trained, but the figures and adata are not saved until the end.
# TODO:  export results to tables.  artifacts are currently:  "figures" and "tables" (to be implimented)

if gen_plots:
    # train
    print(f"archive train plots: {'📈 '*25}")
    archive_plots(
        train_data, model.model[model_name], "train", labels_key=labels_key, path=artifacts_path
    )

    # query
    print(f"archive test plots: {'📊 '*25}")
    archive_plots(
        query_data, model.model[model_name], "query", labels_key=labels_key, path=artifacts_path
    )

# In[ ]
## EXPORT ADATAs ###################################################################
print(f"archive adata: {'💾 '*25}")

if train_data is not None:  # just in case we are only "querying" or "getting"
    archive_data(train_data, output_data_path)
if query_data is not None:
    archive_data(query_data, output_data_path)

# %%
