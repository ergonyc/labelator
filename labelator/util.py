

import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from pathlib import Path




def read_csv_to_coo_matrix(file_path:str|Path, filter_features:list|None = None) -> tuple[coo_matrix, list, list]:
    """ Read a CSV file into a COO matrix
    
    
    returns:
        matrix_coo: the COO matrix
        header: the header of the CSV file
        ids: the ids of the rows of the CSV file
    
    """

    if filter_features is None:
        targets_set = set()
        process_all = True
    else:
        targets_set = set(filter_features)
        process_all = False


    # Initialize lists to hold the COO data
    row_indices = []
    col_indices = []
    data = []
    ids = []
    keep_rows = []

    row_idx = 0

    with open(file_path, 'r') as file:
        print(f"opened {file_path}")
        # Read the header line
        header = file.readline()
        # Read the rest of the lines
        for frow_idx, line in enumerate(file):
            # Get the first entry by reading up to the first comma
            first_entry = line.split(',', 1)[0]
            # Check if the first entry is a target
            if first_entry in targets_set or process_all:
                keep_rows.append(frow_idx)
                # read the entire line
                id,*cnts = line.strip().split(',')
                ids.append(id)
                # Split the line into values and convert them to 8-bit integers
                values = np.array(cnts, dtype=np.uint8)
                
                # Get the non-zero indices and corresponding values
                non_zero_indices = np.nonzero(values)[0]
                non_zero_values = values[non_zero_indices]
                
                # Append the data to the COO format lists
                row_indices.extend([row_idx] * len(non_zero_indices))
                col_indices.extend(non_zero_indices)
                data.extend(non_zero_values)
                # incriment the row index
                row_idx += 1


    # Create the COO matrix
    num_rows = row_idx
    header = header.strip().split(',')
    # remove "genes" from header
    header = header[1:]
    num_cols = len(header)
    matrix_coo = coo_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.uint8)
    
    return matrix_coo, header, ids, keep_rows



def make_anndata_from_bigcsv(big_csv_file_path:str|Path, 
                             filter_features:list|None = None, 
                             meta_obs:pd.DataFrame|None = None, 
                             meta_var:pd.DataFrame|None = None) -> ad.AnnData:
    """ create an anndata object from a large csv file 
    
    returns:
        anndata: the AnnData object
    
    """

    # Call the function with the file path
    sparse_matrix_coo, colnms, ids, kept = read_csv_to_coo_matrix(big_csv_file_path, filter_features=filter_features)

    # Convert the COO matrix to a CSR matrix
    if meta_obs is None:
        # obs_ = obs_.merge(meta_obs, left_index=True, right_index=True)
        meta_obs = pd.DataFrame(colnms, index=colnms, columns=['cell_ids'])
    else:
        meta_obs['cell_ids'] = colnms

    if meta_var is None:
        # var_ = var_.merge(meta_var, left_index=True, right_index=True)
        meta_var = pd.DataFrame({'gene_ids':ids, 'big_idx':kept}, index=ids)
    else:
        meta_var['gene_ids'] = ids
        if filter_features is not None:
            meta_var['big_idx'] = kept


    adat_out = ad.AnnData(X=sparse_matrix_coo.transpose().tocsr(), obs=meta_obs, var=meta_var)
    return adat_out
