# script to execute a full labelator pipeline, 
# load data, initialize model, train model, validate model
#   save model and artifacts (minified adata, etc.)

# command line interface using click to execute labelator pipeline's main function



import labelator as lbltr



import click
import logging
import os
import pandas as pd
import scanpy as sc
import scvi
import sys
from pathlib import Path
from typing import Optional, Union



@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--batch_key", default="batch")
@click.option("--cell_type_key", default="cell_type")
@click.option("--predict_key", default="prediction")
@click.option("--resolution", default=0.5)
@click.option("--use_rep", default="X_pca")
@click.option("--n_epochs", default=400)
@click.option("--n_latent", default=30)
@click.option("--n_hidden", default=128)
@click.option("--n_layers", default=2)
@click.option("--n_neighbors", default=10)
@click.option("--n_pcs", default=50)
@click.option("--n_top_genes", default=2000)
@click.option("--n_top_genes_by_counts", default=2000)


def main():
    """
    Execute labelator pipeline
    """
    logging.basicConfig(level=logging.INFO)

    # load data
    adata = sc.read(input_file)

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




