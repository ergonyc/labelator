import pandas as pd
import numpy as np


def get_stats_table(
    probabilities: dict, categories: np.ndarray, index: np.ndarray, soft: bool = True
) -> pd.DataFrame:
    """
    Create a pandas DataFrame with the probabilities and other stats.

    Parameters
    ----------
    probabilities : dict
        Dictionary of probabilities from `get_stats_from_logits`.
    categories : np.ndarray
        Array of categories.
    index : np.ndarray
        Array of index values corresponding to cell names
    soft : bool
        Whether to return the soft probabilities. Default is `True`.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with the probabilities and other stats.


    """

    if not soft:
        preds_df = pd.DataFrame(probabilities["logit"], columns=categories, index=index)
        preds_df = preds_df.idxmax(axis=1)
    else:
        preds_df = pd.DataFrame(
            probabilities["logit"], columns=[f"lg_{c}" for c in categories], index=index
        )
        for i, c in enumerate(categories):
            preds_df[f"p_{c}"] = probabilities["prob"][:, i]

        preds_df["label"] = probabilities["label"]
        preds_df["max_p"] = probabilities["max_p"]
        preds_df["entropy"] = probabilities["entropy"]
        preds_df["logE"] = probabilities["logE"]
        preds_df["mop"] = probabilities["mop"]

    return preds_df


# def make_latent_adata(vae, **get_latent_args):
#     # TODO: make it accept adata as input
#     #   also probably take the qzv.log() of the latent variables for nicer targets...
#     return_dist = get_latent_args.pop("return_dist", True)

#     if return_dist:
#         qzm, qzv = vae.get_latent_representation(return_dist=return_dist,**get_latent_args)
#         latent_adata = ad.AnnData(np.concatenate([qzm,qzv], axis=1))
#         # latent_adata.obsm["X_latent_qzm"] = qzm
#         # latent_adata.obsm["X_latent_qzv"] = qzv
#         var_names = [f"zm_{i}" for i in range(qzm.shape[1])]+[f"zv_{i}" for i in range(qzv.shape[1])]
#     else:
#         latent_adata = ad.AnnData(vae.get_latent_representation(**get_latent_args))
#         var_names = [f"z_{i}" for i in range(latent_adata.shape[1])]


#     latent_adata.obs_names = vae.adata.obs_names.copy()
#     latent_adata.obs = vae.adata.obs.copy()
#     latent_adata.var_names = var_names
#     return latent_adata


#    adata = add_scanvi_predictions(
#         adata, scanvi_query, insert_key=SCANVI_PREDICTIONS_KEY
#     )
