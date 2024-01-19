import click
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
    VALID_MODEL_NAMES,
    # SCANVI MODELS
    SCANVI_BATCH_EQUALIZED_MODEL_NAME,
    SCANVI_MODEL_NAME,
    # SCVI expression models
    LBL8R_SCVI_EXPRESION_MODEL_NAME,
    XGB_SCVI_EXPRESION_MODEL_NAME,
    # SCVI embedding models
    SCVI_LATENT_MODEL_NAME,
    XGB_SCVI_LATENT_MODEL_NAME,
    # PCS models
    SCVI_EXPR_PC_MODEL_NAME,
    XGB_SCVI_EXPR_PC_MODEL_NAME,
)


def validate_model_name(ctx, param, value):
    valid_name = [v for v in VALID_MODEL_NAMES if v in value]

    if len(valid_name) < 1:
        err_msg = (
            VALID_MODEL_NAMES[0]
            + ", ".join(VALID_MODEL_NAMES[1:-1])
            + ", or "
            + VALID_MODEL_NAMES[-1]
        )
        raise click.BadParameter(f"model_name must be one of {err_msg}")
    elif len(valid_name) > 1:
        print(f"WARNING:  model_name={value} could match to: {': '.join(valid_name)}")
    return value


@click.command()

# model paths / names
@click.option(
    "--model-path",
    type=click.Path(exists=False, path_type=Path),
    required=True,
    help="Path to load/save the trained model.",
)
@click.option(
    "--model-name",
    type=str,
    callback=validate_model_name,
    required=True,
    help=(
        "Name of the model to load/train. Must be one of: "
        + VALID_MODEL_NAMES[0]
        + " "
        + ", ".join(VALID_MODEL_NAMES[1:-1])
        + ", or "
        + VALID_MODEL_NAMES[-1]
    ),
)

# data paths / names
@click.option(
    "--data-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    show_default=True,
    required=False,
    help="Full path to the training data. Will skip to `query` if not provided.",
)
@click.option(
    "--query-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    show_default=True,
    required=False,
    help="Full Path to query data. Will skip query if not provided",
)


# artifacts (figures, data outputs, etc)
@click.option(
    "--output-data-path",
    type=click.Path(writable=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    show_default=True,
    required=False,
    help="Path to save AnnData artifacts.",
)
@click.option(
    "--artifacts-path",
    type=click.Path(writable=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    show_default=True,
    required=False,
    help="""
        Path to save artifacts. Figures will be saved to '`artifacts-path`/figs/`model-name`/' (if `
        make-plots` is True) and results tables will be saved to '`artifacts-path`/results/`model-name`' 
        """,
)
@click.option(
    "--gen-plots/--no-plots",
    is_flag=True,
    default=True,
    show_default=True,
    required=False,
    help="Flag to generate plots.",
)

# Training options
@click.option(
    "--retrain-model",
    is_flag=True,
    help="Flag to force re-training the model. Default will attempt to load model from file",
)
## set labels_key, batch_key,
@click.option(
    "--labels-key",
    type=str,
    default=CELL_TYPE_KEY,
    show_default=True,
    required=False,
    help="Key to adata.obsm 'ground_truth' labels.",
)
@click.option(
    "--batch-key",
    type=str,
    default=None,
    show_default=True,
    required=False,
    help="""
        Key to adata.obsm 'batch' labels for instantiating `scVI`.. e.g. `sample`. Defaults 
        to None wich will instantiate scVI having no batch correction.
        """,
)
#
# TODO:  add options for model configureaion: e.g. n_hidden, n_latent, n_layers, dropout_rate, dispersion, gene_likelihood, latent_distribution encode_covariates,
# TODO: enable other **training_kwargs:  train_size, accelerators, devices, early_stopping, early_stopping_kwargs, batch_size, epochs, etc.


# TODO: add logging
def cli(
    data_path,
    query_path,
    model_path,
    model_name,
    output_data_path,
    artifacts_path,
    gen_plots,
    retrain_model,
    labels_key,
    batch_key,
):
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
            raise click.UsageError(
                "Must provide training data (`data-path`) to retrain model"
            )
        train_data = None

    if query := query_path is not None:
        query_data = load_query_data(query_path)
        # load model with query_data if training data is not provided
        if train_data is None:
            train_data = query_data

    else:
        query_data = None

    if not (train | query):
        raise click.UsageError(
            "Must provide either `data-path` or `query-path` or both"
        )

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

        print(f"query_model: {'🔮 '*25}")
        query_data = query_model(query_data, model)
    # In[ ]
    if train:
        print(f"train_model: {'🏋️ '*25}")
        train_data = query_model(train_data, model)

    # In[ ]
    ## CREATE ARTIFACTS ###################################################################
    # TODO:  wrap in Models, Figures, and Adata in Artifacts class.
    #       currently the models are saved as soon as they are trained, but the figures and adata are not saved until the end.
    # TODO:  export results to tables.  artifacts are currently:  "figures" and "tables" (to be implimented)

    if gen_plots:
        # train
        print(f"archive train plots: {'📈 '*25}")

        archive_plots(
            train_data, model, "train", labels_key=labels_key, path=artifacts_path
        )
        # query
        print(f"archive test plots: {'📊 '*25}")

        archive_plots(
            query_data, model, "query", labels_key=labels_key, path=artifacts_path
        )

    # In[ ]
    ## EXPORT ADATAs ###################################################################
    print(f"archive adata: {'💾 '*25}")

    if train_data is not None:  # just in case we are only "querying" or "getting"
        archive_data(train_data, output_data_path)
    if query_data is not None:
        archive_data(query_data, output_data_path)


if __name__ == "__main__":
    cli()
