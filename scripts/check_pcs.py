
## test equivance of pcs and genes for all models


# In[ ]
import torch
from pathlib import Path

from lbl8r.model.utils._artifact import (
    load_genes,
    load_pcs,
)

import numpy as np

# 5k
# repr models
repr_model_path = "models5k/REPR/scvi"
repr_model_names = ["scvi_emb", "scvi_expr", "scvi_expr_pcs"]
ps = {}
gn = {}
for model_name in repr_model_names:
    model_path = Path(repr_model_path) / model_name
    print(f"{model_name=}")
    pcs = load_pcs(model_path)
    genes = load_genes(model_path)
    print(f"{pcs.shape=} {len(genes)=}")
    # print(f"{pcs[:5]=} {genes[:5]=}")
    gn[model_name] = genes
    ps[model_name] = pcs

# %%
print(f"scvi_emb::scvi_expr - {np.allclose(ps['scvi_emb'], ps['scvi_expr'])}")
print(f"scvi_emb::scvi_expr_pcs - {np.allclose(ps['scvi_emb'], ps['scvi_expr_pcs'])}")
print(f"scvi_expr::scvi_expr_pcs - {np.allclose(ps['scvi_expr'], ps['scvi_expr_pcs'])}")



# In[ ]


# cnt models
count_model_path = "models5k/CNT"  
count_model_names = ["pcs_lbl8r", "raw_lbl8r"]
ps = {}
gn = {}
for model_name in count_model_names:
    model_path = Path(count_model_path) / model_name
    print(f"{model_name=}")
    pcs = load_pcs(model_path)
    genes = load_genes(model_path)
    print(f"{pcs.shape=} {len(genes)=}")
    # print(f"{pcs[:5]=} {genes[:5]=}")
    gn[model_name] = genes
    ps[model_name] = pcs

print(f"pcs_lbl8r::raw_lbl8r - {np.allclose(ps['pcs_lbl8r'], ps['raw_lbl8r'])}")


# In[ ]
# transfer models
trans_model_path = "models5k/TRANSFER/"  
transfer_model_names = ["scanvi_batch_eq", "scanvi"]

ps = {}
gn = {}
for model_name in transfer_model_names:
    model_path = Path(trans_model_path) / model_name
    print(f"{model_name=}")
    pcs = load_pcs(model_path)
    genes = load_genes(model_path)
    print(f"{pcs.shape=} {len(genes)=}")
    # print(f"{pcs[:5]=} {genes[:5]=}")
    gn[model_name] = genes
    ps[model_name] = pcs

print(f"scanvi_batch_eq::scanvi - {np.allclose(ps['scanvi_batch_eq'], ps['scanvi'])}")


# In[ ]



# 10k
model_path="models10k/REPR/scvi"  




# TODO: add logging
train_path = Path("data/scdata/xylena/brain_atlas_anndata_train_cnt.h5ad")
query_path = Path("data/scdata/xylena/brain_atlas_anndata_test_cnt.h5ad")
train_path = Path("data/scdata/xylena5k/xyl2_train.h5ad")
query_path = Path("data/scdata/xylena5k/xyl2_test.h5ad")
query_path = Path("data/scdata/xylena5k/xyl2_query.h5ad")

# train_path = None
# train_path = None
# query_path = Path('data/scdata/ASAP/artifacts/06_merged_filtered_processed_integrated_clustered_anndata_object.h5ad')
# # query_path = Path('data/scdata/ASAP/artifacts/06_merged_filtered_integrated_clustered_anndata_object.h5ad')
# query_path = Path('data/scdata/ASAP/artifacts/07_merged_filtered_integrated_clustered_annotated_anndata_object.h5ad')
# model_path = Path("models/CNT2/")
model_path = Path("models5k/REPR/scvi/")
# model_path = Path("models/TRANSFER2/")
# train_path = None
# model_name = "raw_lbl8r"
# model_name = "scvi_emb_xgb"
model_name = "pcs_lbl8r"
# model_name = "scvi_emb"
# model_name = "scanvi_batch_eq"

output_data_path = Path("data/scdata/xylena5k/LABELATOR/")
artifacts_path = Path("artifacts5k/")
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
# %%
