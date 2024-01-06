import torch.nn.functional as F
from torch.distributions import Categorical
import torch
import pandas as pd
import numpy as np
from anndata import AnnData
from scvi.model import SCANVI

# from scanpy.pp import pca


def calculate_margin_of_probability(probabilities: torch.Tensor) -> np.ndarray:
    """
    Calculate the margin of probability.

    Parameters
    ----------
    probabilities : torch.Tensor
        Probabilities from a model.

    Returns
    -------
    np.ndarray
        Array of margin of probability.
    """
    # Get the top two probabilities
    top_probs, _ = torch.topk(probabilities, 2)

    # Calculate the margin
    margin = top_probs[:, 0] - top_probs[:, 1]
    return margin.numpy()


def get_stats_from_logits(logits: torch.Tensor, categories: np.ndarray) -> dict:
    """
    Get probabilities, entropy, log entropy, labels, and margin of probability from logits.

    Parameters
    ----------
    logits : torch.Tensor
        Logits from a model.
    categories : np.ndarray
        Array of categories.

    Returns
    -------
    dict
        Dictionary of probabilities, entropy, log entropy, labels, and margin of probability.

    """
    # Applying softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)

    # Create a Categorical distribution
    distribution = Categorical(probs=probabilities)

    # Calculate entropy
    entropy = distribution.entropy()

    # n_classes = probabilities.shape[1]

    logs = logits.numpy()
    probs = probabilities.numpy()
    ents = entropy.numpy()
    logents = entropy.log().numpy()

    # print("Logits: ", logs)
    # print("Probabilities: ", probs)

    maxprobs = probs.max(axis=1)

    labels = categories[probs.argmax(axis=1)]

    margin = calculate_margin_of_probability(probabilities)

    return {
        "logit": logs,
        "prob": probs,
        "entropy": ents,
        "logE": logents,
        "max_p": maxprobs,
        "mop": margin,
        "label": labels,
    }


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


def add_scanvi_predictions(ad: AnnData, model: SCANVI, insert_key: str = "label"):
    """
    Get the "soft" and label predictions from a SCANVI model,
    and then add into the ad.obs

    Parameters
    ----------
    ad : ad.AnnData
        AnnData object to add the predictions to
    model : SCANVI
        SCANVI model to use to get the predictions
    Returns
    -------
    ad.AnnData
        AnnData object with the predictions added

    """

    predictions = model.predict(ad, soft=True)
    predictions[insert_key] = model.predict(ad, soft=False)

    obs = ad.obs
    if set(predictions.columns) & set(obs.columns):
        ValueError("Predictions and obs have overlapping columns")
        return ad

    obs = pd.merge(obs, predictions, left_index=True, right_index=True, how="left")

    ad.obs = obs
    return ad
