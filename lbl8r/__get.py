## wrappers for getting trained models

from lbl8r import (
    get_lbl8r,
    query_lbl8r,
    get_lbl8r_scvi,
    get_trained_scvi,
    get_trained_scanvi,
)
from lbl8r.constants import *
from lbl8r.constants import XYLENA_PATH


def get_model(
    adata,
    model_name,
    mode,
    model_path,
    label_key,
    retrain=False,
    plot_training=False,
    **fig_kwargs,
):
    """
    Get a model.
    """
    if model_name == "scvi":
        model = get_trained_scvi(adata, mode, model_path)

        vae_model_name = "scvi"
        (
            vae,
            train_ad,
        ) = get_lbl8r_scvi(  # sa,e ast get_trained_scvi but forces "batch"=None
            train_ad,
            labels_key=label_key,
            model_path=model_path,
            retrain=retrain,
            model_name=vae_model_name,
            plot_training=plot_training,
            **fig_kwargs,
        )

    elif model_name == "lbl8r":
        model = get_trained_lbl8r(data, mode, model_path)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model
