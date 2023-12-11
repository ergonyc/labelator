

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from pathlib import Path





def add_nn_umap(adata: ad.AnnData,
                use_rep:str,
                resolution:float = 0.5) -> None:
    """
    Compute umaps of the QC data
    """

    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.leiden(adata, resolution=resolution)

    sc.tl.umap(adata)






def plot_umaps(adata: ad.AnnData, 
                  predict_key:str,
                  annot:str = None, 
                  ) -> None:
    """
    Plot umaps of the QC data
    """
    sc.pl.umap(
        adata,
        color=["batch", "leiden"],
        frameon=False,
        ncols=1,
        save=f"{annot}-batch-leiden.png"
    )    

    sc.pl.umap(
        adata,
        color=["sample"],
        frameon=False,
        ncols=1,
        save=f"{annot}-sample.png"
    )

    sc.pl.umap(
        adata,
        color= ["cell_type",predict_key],
        frameon=False,
        ncols=1,
        save=f"{annot}-cell_type-prediction.png"
        )

    sc.pl.umap(
        adata,
        color= ["doublet_score", "percent.mt", "percent.rb"],
        frameon=False,
        ncols=1,
        save=f"{annot}-noise.png"
    )