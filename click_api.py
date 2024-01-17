import click
import torch

from lbl8r.labelator import (
    setup_paths,
    load_training_data,
    load_query_data,
    prep_pc_data,
    prep_latent_data,
    get_model,
    query_model,
    create_artifacts,
    CELL_TYPE_KEY,
)


@click.command()

# model paths / names
@click.option(
    "--model-path",
    type=click.Path(exists=False),
    require=True,
    help="Path to load/save the trained model.",
)
@click.option(
    "--model-name", type=str, require=True, help="Name of the model to load/train."
)

# data paths / names
@click.option(
    "--data-path",
    type=click.Path(exists=True),
    default=None,
    show_default=True,
    required=False,
    help="Full path to the training dataset. Will skip to `query` if not provided.",
)
@click.option(
    "--query-path",
    type=click.Path(exists=True),
    default=None,
    show_default=True,
    required=False,
    help="Full Path to query data. Will skip query if not provided",
)


# artifacts (figures, data outputs, etc)
@click.option(
    "--output-data-path",
    type=click.Path(exists=True),
    default=None,
    show_default=True,
    required=False,
    help="Path to save AnnData artifacts.",
)
@click.option(
    "--artifacts-path",
    type == click.Path(exists=True),
    default=None,
    show_default=True,
    required=False,
    help="""
        Path to save artifacts. Figures will be saved to '`artifacts-path`/figs/`model-name`/' (if `
        make-plots` is True) and results tables will be saved to '`artifacts-path`/results/`model-name`' 
        """,
)
@click.option(
    "--make-plots",
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
    default=False,
    show_default=True,
    required=False,
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
    make_plots,
    retrain_model,
):
    """
    Command line interface for model processing pipeline.
    """

    # setup
    torch.set_float32_matmul_precision("medium")

    # load data
    if train := data_path is not None:
        train_data = load_training_data(data_path)

    if query := query_path is not None:
        query_data = load_training_data(query_path)

    if not (train | query):
        raise click.UsageError(
            "Must provide either `data-path` or `query-path` or both"
        )

    data = load_and_prep(data_path)
    if train_model:
        model = get_model(data, model_name, "train", save_model_path)
    else:
        model = get_model(None, model_name, "load", load_model_path)

    ## GET   ###################################################################
    model = get_model(
        train_data,
        model_name=model_name,
        model_path=model_path,
        labels_key=cell_type_key,
        retrain=retrain_model,
        **training_kwargs,
    )

    if query:
        query_data = load_and_prep(query_data_path)
        query_results = query_model(query_data, model)
        with open(output_path, "w") as file:
            file.write(str(query_results))

    if artifacts:
        create_artifacts(
            visualization_path=visualization_path if generate_visualizations else None,
            artifacts_path=artifacts_path if generate_artifacts else None,
        )


if __name__ == "__main__":
    cli()


# python myscript.py --setup --data-path /path/to/data --model --model-name MyModel --train-model --query --query-data-path /path/to/query/data --output-path /path/to/output --artifacts --generate-visualizations --visualization-path /path/to/visualizations
