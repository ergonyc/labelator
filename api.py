import argparse
from mymodule import (
    setup_paths,
    load_and_prep,
    get_model,
    load_query_data,
    query_model,
    create_artifacts,
)


def main():
    parser = argparse.ArgumentParser(
        description="Command line interface for the model processing pipeline."
    )

    # Setup arguments
    parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model."
    )
    parser.add_argument(
        "--model-params-file",
        type=str,
        help="Path to a file containing model parameters.",
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--config-path", type=str, help="Path to additional configuration files."
    )

    # Get Model arguments
    parser.add_argument(
        "--train-model", action="store_true", help="Flag to train the model."
    )
    parser.add_argument(
        "--save-model-path", type=str, help="Path to save the trained model."
    )
    parser.add_argument(
        "--load-model-path", type=str, help="Path to load the model from disk."
    )

    # Query Model arguments
    parser.add_argument(
        "--query-data-path",
        type=str,
        required=True,
        help="Path to the data for making inferences.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the inference results.",
    )

    # Artifacts arguments
    parser.add_argument(
        "--generate-visualizations",
        action="store_true",
        help="Flag to generate visualizations.",
    )
    parser.add_argument(
        "--visualization-path", type=str, help="Path to save visualizations."
    )
    parser.add_argument(
        "--generate-artifacts",
        action="store_true",
        help="Flag to generate data artifacts.",
    )
    parser.add_argument("--artifacts-path", type=str, help="Path to save artifacts.")

    args = parser.parse_args()

    # Part 1: Setup
    setup_args = {
        "model_params_file": args.model_params_file,
        "data_path": args.data_path,
        "config_path": args.config_path,
    }
    setup_paths(**setup_args)

    # Part 2: Get Model
    if args.train_model:
        data = load_and_prep(args.data_path)
        model = get_model(data, args.model_name, "train", args.save_model_path)
    else:
        model = get_model(None, args.model_name, "load", args.load_model_path)

    # Part 3: Query Model
    query_data = load_query_data(args.query_data_path)
    query_results = query_model(query_data, model)
    # Assuming you want to save the results of the query
    with open(args.output_path, "w") as file:
        file.write(str(query_results))

    # Part 4: Artifacts
    if args.generate_visualizations or args.generate_artifacts:
        artifacts_args = {
            "visualization_path": args.visualization_path
            if args.generate_visualizations
            else None,
            "artifacts_path": args.artifacts_path if args.generate_artifacts else None,
        }
        create_artifacts(**artifacts_args)


if __name__ == "__main__":
    main()


import click
from mymodule import (
    setup_paths,
    load_and_prep,
    get_model,
    load_query_data,
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
        query_data = load_query_data(query_data_path)
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