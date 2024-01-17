import argparse
from mymodule import (
    setup_paths,
    load_and_prep,
    get_model,
    load_query_data,
    query_model,
    create_artifacts,
)


def setup_cli(subparsers):
    setup_parser = subparsers.add_parser(
        "setup", help="Setup paths and configurations."
    )
    setup_parser.add_argument(
        "--model-params-file",
        type=str,
        help="Path to a file containing model parameters.",
    )
    setup_parser.add_argument(
        "--data-path", type=str, required=True, help="Path to the dataset."
    )
    setup_parser.add_argument(
        "--config-path", type=str, help="Path to additional configuration files."
    )
    setup_parser.set_defaults(func=setup_command)


def get_model_cli(subparsers):
    model_parser = subparsers.add_parser("model", help="Load or train a model.")
    model_parser.add_argument(
        "--model-name", type=str, required=True, help="Name of the model."
    )
    model_parser.add_argument(
        "--train-model", action="store_true", help="Flag to train the model."
    )
    model_parser.add_argument(
        "--save-model-path", type=str, help="Path to save the trained model."
    )
    model_parser.add_argument(
        "--load-model-path", type=str, help="Path to load the model from disk."
    )
    model_parser.set_defaults(func=model_command)


def query_cli(subparsers):
    query_parser = subparsers.add_parser("query", help="Query the model with data.")
    query_parser.add_argument(
        "--query-data-path",
        type=str,
        required=True,
        help="Path to the data for making inferences.",
    )
    query_parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the inference results.",
    )
    query_parser.set_defaults(func=query_command)


def artifacts_cli(subparsers):
    artifacts_parser = subparsers.add_parser(
        "artifacts", help="Generate artifacts and visualizations."
    )
    artifacts_parser.add_argument(
        "--generate-visualizations",
        action="store_true",
        help="Flag to generate visualizations.",
    )
    artifacts_parser.add_argument(
        "--visualization-path", type=str, help="Path to save visualizations."
    )
    artifacts_parser.add_argument(
        "--generate-artifacts",
        action="store_true",
        help="Flag to generate data artifacts.",
    )
    artifacts_parser.add_argument(
        "--artifacts-path", type=str, help="Path to save artifacts."
    )
    artifacts_parser.set_defaults(func=artifacts_command)


def setup_command(args):
    setup_paths(
        model_params_file=args.model_params_file,
        data_path=args.data_path,
        config_path=args.config_path,
    )


def model_command(args):
    data = load_and_prep(args.data_path)
    if args.train_model:
        model = get_model(data, args.model_name, "train", args.save_model_path)
    else:
        model = get_model(None, args.model_name, "load", args.load_model_path)


def query_command(args):
    query_data = load_query_data(args.query_data_path)
    query_results = query_model(query_data, model)
    with open(args.output_path, "w") as file:
        file.write(str(query_results))


def artifacts_command(args):
    create_artifacts(
        visualization_path=args.visualization_path
        if args.generate_visualizations
        else None,
        artifacts_path=args.artifacts_path if args.generate_artifacts else None,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Command line interface for model processing pipeline."
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")
    subparsers.required = True

    setup_cli(subparsers)
    get_model_cli(subparsers)
    query_cli(subparsers)
    artifacts_cli(subparsers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
