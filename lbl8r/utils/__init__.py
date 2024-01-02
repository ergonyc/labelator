from ._mde import mde
from ._pred import ( 
                get_stats_from_logits, 
                get_stats_table, 
                make_latent_adata, 
                make_pc_loading_adata, 
                make_scvi_normalized_adata,
                add_predictions_to_adata,
                )
from ._plot import plot_embedding, plot_predictions

__all__ = [
    "mde",
    "get_stats_from_logits",
    "get_stats_table",
    "make_latent_adata",
    "make_pc_loading_adata",
    "make_scvi_normalized_adata",
    "plot_embedding",
    "add_predictions_to_adata",
    "plot_predictions"
]
