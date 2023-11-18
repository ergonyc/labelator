


import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from pathlib import Path
import scanpy as sc
import numpy as np


from .util import make_anndata_from_bigcsv, load_10x_tar_gz



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


    # sc.external.pp.scrublet(adata, batch_key="sample")


    return adata


def convert_to_h5ad(data_path: str|Path,
                             filter_features:list|None = None, 
                             remake:bool=False) -> ad.AnnData:
    """
    convert .h5 files to .h5ad files. 
    We will save them as "full" files, meaning that we will include all cells, but will add a column to the obs
    that indicates whether the cell was filtered out or not by the cellranger pipeline.
    also add 'sample_id' column to obs corresponding to the filename sample_id
    
    
    """

    # get list of tar.gz files in data_path
    # tar_gz_files = [f for f in data_path.iterdir() if f.suffix == ".gz"]
    filt_feature_files = list(data_path.glob("*.filtered_feature_bc_matrix.h5"))


    for file_n in filt_feature_files:
        

        sample_id = file_n.stem.split(".")[0]
        out_name = f"{sample_id}_full.h5ad"
        if Path(data_path.parent / out_name).exists() and not remake:
            print(f"skipping {file_n}")
        else:

            ad_filt = sc.read_10x_h5(file_n)
            raw_file_n = str(file_n).replace('filtered_','raw_')
            ad_raw= sc.read_10x_h5(raw_file_n)
    
            ad_raw.var_names_make_unique()
            ad_raw.obs_names_make_unique()
            filtered_cells = ad_filt.obs_names
            ad_raw.obs['filtered_cells'] = ad_raw.obs_names.isin(filtered_cells)
            ad_raw.obs['sample_id'] = sample_id

            X = ad_raw.X
            ad_raw.X = csr_matrix(X, dtype=np.uint8)
            ad_raw.write_h5ad(data_path.parent / out_name)


def convert_jakobsson_data_from_tar_10x(data_path: str|Path,
                             filter_features:list|None = None, 
                             remake:bool=False) -> ad.AnnData:
    """
    convert
    
    """

    # get list of tar.gz files in data_path
    # tar_gz_files = [f for f in data_path.iterdir() if f.suffix == ".gz"]
    tar_gz_files = list(data_path.glob('*.tar.gz'))
    for tar_gz in tar_gz_files:
        print(tar_gz)
        # read the data
        adata = load_10x_tar_gz(tar_gz)
         
        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        X = adata.X
        X = csr_matrix(X, dtype=np.uint8)
        adata.X = X
        nm_parts = tar_gz.stem.split('_')
        out_name = f"{nm_parts[0]}_{nm_parts[1]}.h5ad"
        adata.write_h5ad(data_path.parent / out_name)

    


def standard_qc(ad_inb:ad.AnnData) -> ad.AnnData:

    """
    perform "standard" qc on anndata object

    """
    pass 

