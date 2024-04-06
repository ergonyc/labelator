import click
import torch
from pathlib import Path

from lbl8r.labelator import (
    load_data,
    query_model,
    prep_query_model,
    archive_data,
    archive_plots,
    load_trained_model,
    CELL_TYPE_KEY,
    VALID_MODEL_NAMES,
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
    "--train-path",
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

# TODO:  add options for model configureaion: e.g. n_hidden, n_latent, n_layers, dropout_rate, dispersion, gene_likelihood, latent_distribution encode_covariates,
# TODO: enable other **training_kwargs:  train_size, accelerators, devices, early_stopping, early_stopping_kwargs, batch_size, epochs, etc.


# TODO: add logging
def cli(
    train_path,
    query_path,
    model_path,
    model_name,
    output_data_path,
    artifacts_path,
    gen_plots,
    retrain_model,
    labels_key,
    # batch_key,
):
    """
    Command line interface for model processing pipeline.
    """
    print(f"{train_path=}:: {query_path=}:: {model_path=}:: {model_name=}")
    print(
        f"{output_data_path=}:: {artifacts_path=}:: {gen_plots=}:: {retrain_model=}:: {labels_key=}"
    )
    # setup
    torch.set_float32_matmul_precision("medium")

    ## LOAD DATA ###################################################################
    query_data = load_data(query_path, archive_path=output_data_path)

    ## PREP MODEL ###################################################################
    # gets model and preps Adata
    # TODO:  add additional training_kwargs to cli
    training_kwargs = {}  # dict(batch_key=batch_key)
    print(f"prep_model: {'🛠️ '*25}")

    model_set = load_trained_model(model_name, model_path, labels_key=labels_key)
    # if no traing data is loaded (just prepping for query) return placeholder data

    # In[ ]
    ## QUERY MODELs ###################################################################
    # makes sure the genes correspond to those of the prepped model
    #     projects counts onto the principle components of the training datas eigenvectors as 'X_pca'
    # TODO:  add additional training_kwargs to cli
    print(f"prep query: {'💅 '*25}")
    # prep query model actually preps data unless its a scANVI model...
    #
    model_set, query_data = prep_query_model(
        query_data,
        model_set,
        model_name,
        labels_key=labels_key,
        retrain=retrain_model,
    )

    # In[ ]
    print(f"query_model: {'🔮 '*25}")
    query_data = query_model(query_data, model_set)
    # In[ ]
    ## CREATE ARTIFACTS ###################################################################
    if gen_plots:
        print(f"archive query plots and data: {'📊 '*25}")
        archive_plots(
            query_data, model_set, "query", fig_path=(artifacts_path / "figs")
        )

    print(f"archive query plots and data: {'📊 '*25}")
    archive_plots(query_data, model_set, "query", fig_path=(artifacts_path / "figs"))
    print(f"archive query output adata: {'💾 '*25}")
    archive_data(query_data)

    # In[ ]


if __name__ == "__main__":
    cli()
