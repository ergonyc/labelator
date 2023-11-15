


import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from pathlib import Path
import scanpy as sc
import numpy as np


from .util import make_anndata_from_bigcsv



XYLENA_RAW_CSV = "brain_atlas_full_counts_table.csv"
XYLENA_RAW_H5AD = "brain_atlas_anndata.h5ad"
XYLENA_TRAINING_SET = "Model Combinations - training_set_98.csv"
XYLENA_CLEAN_SAMPLES = "Model Combinations - clean_samples_138.csv"
XYLENA_OBS = "cell_barcode_labels.csv"
XYLENA_PATH = "xylena_raw"

def get_xylena_data_from_raw(data_path: str|Path,
                             filter_features:list|None = None, 
                             remake:bool=False) -> ad.AnnData:
    """
    reads raw data from the xylena_raw dataset into an AnnData object

    """

    # try to load from h5ad unless remake is True
    if not remake:
        try:
            return ad.read_h5ad(data_path / XYLENA_RAW_H5AD)
        except:
            pass


    obs = pd.read_csv(data_path / XYLENA_OBS, index_col=0)

    # read the data
    adata = make_anndata_from_bigcsv(data_path / XYLENA_RAW_CSV, 
                                    filter_features=filter_features, 
                                    meta_obs=obs)

    clean_samples = pd.read_csv(data_path / XYLENA_CLEAN_SAMPLES)

    batch_mapper = dict(zip(clean_samples["sample"], clean_samples["batch"]))
    adata.obs["batch"] = adata.obs["sample"].map(batch_mapper)

    test_samples = pd.read_csv(data_path / XYLENA_TRAINING_SET)
    adata.obs["training"] = [s in test_samples['sample'].values for s in adata.obs["sample"]]

    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")  # "MT-" for human, "Mt-" for mouse
    # ribosomal genes
    adata.var["rb"] = adata.var_names.str.startswith(("RPS", "RPL"))


    sc.external.pp.scrublet(adata, batch_key="sample")


    return adata



def standard_qc(ad_inb:ad.AnnData) -> ad.AnnData:

    """
    perform "standard" qc on anndata object

    """
    pass 



