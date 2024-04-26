import click
import torch
from pathlib import Path

from lbl8r.labelator import (
    train_lbl8r,
    # load_data,
    # prep_model,
    # query_model,
    # get_trained_model,
    # archive_plots,
    # archive_data,
    CELL_TYPE_KEY,
    VALID_MODEL_NAMES,
)

# setup
torch.set_float32_matmul_precision("medium")


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
def cli(
    train_path,
    model_path,
    model_name,
    output_data_path,
    artifacts_path,
    gen_plots,
    retrain_model,
    labels_key,
):
    """
    Command line interface for model training processing pipeline.
    """
    train_lbl8r(
        train_path,
        model_path,
        model_name,
        output_data_path,
        artifacts_path,
        gen_plots,
        retrain_model,
        labels_key,
    )


if __name__ == "__main__":
    cli()
