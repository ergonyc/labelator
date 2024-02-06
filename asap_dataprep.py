#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python
# coding: utf-8

### set up train and test AnnData objects for LBL8R

# In[ ]:
### import local python functions in ../lbl8r
import sys
import os

import scanpy as sc
import anndata as ad
from pathlib import Path
import scipy.sparse as sp
import pandas as pd


train_path = Path("data/scdata/xylena/brain_atlas_anndata_train_cnt.h5ad")


query_path = Path(
    "data/scdata/ASAP/artifacts/06_merged_filtered_processed_integrated_clustered_anndata_object.h5ad"
)
query_path = Path(
    "data/scdata/ASAP/artifacts/06_merged_filtered_integrated_clustered_anndata_object.h5ad"
)

# load train data for reference
train_ad = ad.read_h5ad(train_path)

asap_ad = ad.read_h5ad(query_path)


# In[ ]:
train_ad.write_h5ad(train_filen)


# In[ ]:
