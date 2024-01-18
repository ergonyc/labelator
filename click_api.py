import click
import torch

from lbl8r.labelator import (
    setup_paths,
    load_training_data,
    load_query_data,
    prep_pc_data,
    prep_latent_data,
    prep_expr_data,
    get_model,
    query_model,
    create_artifacts,
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
    type=click.Path(exists=False),
    require=True,
    help="Path to load/save the trained model.",
)
@click.option(
    "--model-name",
    type=str,
    callback=validate_model_name,
    require=True,
    help=(
        "Name of the model to load/train. Must be one of: "
        + VALID_MODEL_NAMES[0]
        + ", ".join(VALID_MODEL_NAMES[1:-1])
        + ", or "
        + VALID_MODEL_NAMES[-1]
    ),
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
        query_data = load_training_data(query_path)
        # load model with query_data if training data is not provided
        if train_data is None:
            train_data = query_data

    else:
        query_data = None

    if not (train | query):
        raise click.UsageError(
            "Must provide either `data-path` or `query-path` or both"
        )

    ## GET MODEL ###################################################################
    # TODO:  add additional training_kwargs to cli
    training_kwargs = dict(batch_key=batch_key)
    model = get_model(
        train_data,  # Note this is actually query_data if train_data arg was None
        model_name=model_name,
        model_path=model_path,
        labels_key=labels_key,
        retrain=retrain_model,
        **training_kwargs,
    )

    # # unpack the vae if it is a tuple (scanvi or scvi model or derivatives)
    # if isinstance(model, tuple):
    #     assert model_name in (
    #         SCANVI_BATCH_EQUALIZED_MODEL_NAME,
    #         SCANVI_MODEL_NAME,
    #         LBL8R_SCVI_EXPRESION_MODEL_NAME,
    #         XGB_SCVI_EXPRESION_MODEL_NAME,
    #         SCVI_LATENT_MODEL_NAME,
    #         XGB_SCVI_LATENT_MODEL_NAME,
    #         SCVI_EXPR_PC_MODEL_NAME,
    #         XGB_SCVI_EXPR_PC_MODEL_NAME,
    #     )
    #     model, vae = model
    # else:
    #     vae = None

    ## QUERY MODEL ###################################################################
    # TODO:  add additional training_kwargs to cli
    if query:
        # 1. prep query data

        if model_name in (
            LBL8R_SCVI_EXPRESION_MODEL_NAME,
            XGB_SCVI_EXPRESION_MODEL_NAME,
        ):
            # SCVI expression models
            query_data = prep_expr_data(query_data, model)

        elif model_name in (
            SCVI_LATENT_MODEL_NAME,
            XGB_SCVI_LATENT_MODEL_NAME,
        ):
            # SCVI embedding models
            query_data = prep_latent_data(query_data, model)

        elif model_name in (
            SCVI_EXPR_PC_MODEL_NAME,
            XGB_SCVI_EXPR_PC_MODEL_NAME,
        ):
            # PCS models
            query_data = prep_pc_data(query_data, model)
        # 2. query model
        # if model_name in (
        #     SCANVI_BATCH_EQUALIZED_MODEL_NAME,
        #     SCANVI_MODEL_NAME,
        # ):
        #     # SCANVI models
        #     query_data = prep_and_query_scanvi(query_data, model)

        query_data = query_model(query_data, model.model, model_name=model_name)
        if isinstance(query_data, tuple):
            assert len(query_data) == 3
            assert model_name in (
                SCANVI_BATCH_EQUALIZED_MODEL_NAME,
                SCANVI_MODEL_NAME,
            )

            query_data, query_data, query_model = query_data

    ## CREATE ARTIFACTS ###################################################################

    if artifacts_path is not None:
        # if model_name =

        if model_name is None:
            data_artifacts = [
                train_data,
                query_data,
            ]
            figure_artifacts = []
        artifacts = [data_artifacts, figure_artifacts]

        if make_plots:
            create_artifacts(
                visualization_path=visualization_path
                if generate_visualizations
                else None,
                artifacts_path=artifacts_path if generate_artifacts else None,
            )


if __name__ == "__main__":
    cli()


# python myscript.py --setup --data-path /path/to/data --model --model-name MyModel --train-model --query --query-data-path /path/to/query/data --output-path /path/to/output --artifacts --generate-visualizations --visualization-path /path/to/visualizations
