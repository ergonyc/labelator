from ._mde import mde

from ._pred import (
    get_stats_from_logits,
    get_stats_table,
    add_scanvi_predictions,
)

from ._adata import (
    make_latent_adata,
    make_pc_loading_adata,
    make_scvi_normalized_adata,
    add_predictions_to_adata,
    add_cols_into_obs,
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
    "get_stats_from_logits",
    "get_stats_table",
    "make_latent_adata",
    "make_pc_loading_adata",
    "make_scvi_normalized_adata",
    "plot_embedding",
    "add_predictions_to_adata",
    "add_cols_into_obs",
    "plot_predictions",
    "add_scanvi_predictions",
    "transfer_pcs",
    "sparsify_adata",
    "export_ouput_adata",
    "plot_scvi_training",
    "plot_scanvi_training",
    "plot_lbl8r_training",
]
