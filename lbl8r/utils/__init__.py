from ._mde import mde

from ._pred import (
    get_stats_table,
)

from ._adata import (
    prep_lbl8r_adata,
    make_latent_adata,
    make_pc_loading_adata,
    make_scvi_normalized_adata,
    add_predictions_to_adata,
    merge_into_obs,
    transfer_pcs,
    sparsify_adata,
    export_ouput_adata,
)
from ._plot import (
    plot_embedding,
    plot_predictions,
    plot_scvi_training,
    plot_scanvi_training,
    plot_lbl8r_training,
)

__all__ = [
    "mde",
    "get_stats_table",
    "prep_lbl8r_adata",
    "make_latent_adata",
    "make_pc_loading_adata",
    "make_scvi_normalized_adata",
    "plot_embedding",
    "add_predictions_to_adata",
    "merge_into_obs",
    "plot_predictions",
    "query_scanvi",
    "transfer_pcs",
    "sparsify_adata",
    "export_ouput_adata",
    "plot_scvi_training",
    "plot_scanvi_training",
    "plot_lbl8r_training",
]
