import click


from lbl8r.labelator import (
    setup_paths,
    load_and_prep,
    get_model,
    query_model,
    create_artifacts,
)


@click.command()
@click.option(
    "--setup/--no-setup", default=False, help="Enable/Disable the setup process."
)
@click.option(
    "--model-params-file", type=str, help="Path to a file containing model parameters."
)
@click.option("--data-path", type=str, help="Path to the dataset.")
@click.option("--config-path", type=str, help="Path to additional configuration files.")
@click.option(
    "--model/--no-model",
    default=False,
    help="Enable/Disable the model loading/training process.",
)
@click.option("--model-name", type=str, help="Name of the model.")
@click.option("--train-model", is_flag=True, help="Flag to train the model.")
@click.option("--save-model-path", type=str, help="Path to save the trained model.")
@click.option("--load-model-path", type=str, help="Path to load the model from disk.")
@click.option(
    "--query/--no-query", default=False, help="Enable/Disable the query process."
)
@click.option(
    "--query-data-path", type=str, help="Path to the data for making inferences."
)
@click.option("--output-path", type=str, help="Path to save the inference results.")
@click.option(
    "--artifacts/--no-artifacts",
    default=False,
    help="Enable/Disable the artifacts generation process.",
)
@click.option(
    "--generate-visualizations", is_flag=True, help="Flag to generate visualizations."
)
@click.option("--visualization-path", type=str, help="Path to save visualizations.")
@click.option(
    "--generate-artifacts", is_flag=True, help="Flag to generate data artifacts."
)
@click.option("--artifacts-path", type=str, help="Path to save artifacts.")
def cli(
    setup,
    model_params_file,
    data_path,
    config_path,
    model,
    model_name,
    train_model,
    save_model_path,
    load_model_path,
    query,
    query_data_path,
    output_path,
    artifacts,
    generate_visualizations,
    visualization_path,
    generate_artifacts,
    artifacts_path,
):
    """
    Command line interface for model processing pipeline.
    """
    if setup:
        setup_paths(
            model_params_file=model_params_file,
            data_path=data_path,
            config_path=config_path,
        )

    if model:
        data = load_and_prep(data_path)
        if train_model:
            model = get_model(data, model_name, "train", save_model_path)
        else:
            model = get_model(None, model_name, "load", load_model_path)

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
