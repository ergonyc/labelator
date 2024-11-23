import pandas as pd
import numpy as np

from ..._constants import UNLABELED


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
    # remove unknown category
    categories = categories[categories != UNLABELED]

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
