# script to execute a full labelator pipeline,
# load data, initialize model, train model, validate model
#   save model and artifacts (minified adata, etc.)

# command line interface using click to execute labelator pipeline's main function


import labelator as lbltr


import click
import logging
import os
import pandas as pd
import anndata as ad 
import sys
from pathlib import Path
from typing import Optional, Union


@click.command()
# anndata paths
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--X_type", default="counts")
# model paths
@click.argument("model_path", type=click.Path())
@click.option("--model_type", default="LBL8R")  # SCVI
@click.option("--model_name", default="LBL8R")
@click.option("--classifier", default="classifier")  # end_to_end, xgb
@click.option("--cell_type_key", default="cell_type")
@click.option("--model_type", default="LBL8R")
@click.option("--batch_key", default="batch")

# model_kwargs
@click.option("--n_epochs", default=200)
@click.option("--n_latent", default=10)
@click.option("--n_hidden", default=128)
@click.option("--n_layers", default=2)
def main():
    """
    Execute labelator pipeline
    """
    logging.basicConfig(level=logging.INFO)

    # load data
    adata = ad.read_h5ad(input_file)

    # initialize model
    model = scvi.model.SCVI(adata)

    # train model
    model.train()

    # validate model
    model.validate()

    # save model and artifacts
    model.save(output_file)


if __name__ == "__main__":
    main()
