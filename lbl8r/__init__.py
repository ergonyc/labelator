from ._load import load_query_data, load_training_data


from ._get import (
    get_lbl8r_scvi,
    get_pca_lbl8r,
    get_trained_scanvi,
    get_trained_scvi,
    get_query_scvi,
    get_query_scanvi,
    get_pca_lbl8r,
    get_lbl8r,
    get_xgb,
    get_model,
)

from ._load import (
    prep_lbl8r_adata,
    # make_pc_loading_adata,
    # load_query_data,
    # load_training_data,
)

from ._query import (
    query_lbl8r,
    query_xgb,
    query_scanvi,
    query_scanvi,
    query_model,
)


__all__ = [
    "get_lbl8r_scvi",
    "get_pca_lbl8r",
    "get_trained_scanvi",
    "get_trained_scvi",
    "get_query_scvi",
    "get_query_scanvi",
    "get_pca_lbl8r",
    "get_lbl8r",
    "get_xgb",
    "get_model",
    # "make_pc_loading_adata",
    # "load_query_data",
    # "load_training_data",
    "query_lbl8r",
    "query_xgb",
    "query_scanvi",
    "query_scanvi",
    "query_model",
]
