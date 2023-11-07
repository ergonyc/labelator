


import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from pathlib import Path


from .util import make_anndata_from_bigcsv


XYLENA_RAW_CSV = "brain_atlas_full_counts_table.csv"
XYLENA_RAW_H5AD = "brain_atlas_anndata.h5ad"
XYLENA_TRAINING_SET = "Model Combinations - training_set_98.csv"


def get_xylena_data_from_raw(root_path: str|Path,
                             filter_features:list|None = None, 
                             remake:bool=False) -> ad.AnnData:
    """
    reads raw data from the xylena_raw dataset into an AnnData object

    """

    # try to load from h5ad unless remake is True
    if not remake:
        try:
            return ad.read_h5ad(root_path / XYLENA_RAW_H5AD)
        except:
            pass


    # read the data
    data = make_anndata_from_bigcsv(root_path / "brain_atlas_full_counts_table.csv", filter_features=filter_features)


def standard_qc(ad_inb:ad.AnnData) -> ad.AnnData:

    """
    perform "standard" qc on anndata object

    """
    pass 



