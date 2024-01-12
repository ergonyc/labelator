### run AFTER lbl8r_scvi.py

# In[ ]:
import scanpy as sc
import anndata as ad
from lbl8r.utils import transfer_pcs, export_ouput_adata
from pathlib import Path
### import local python functions in ../lbl8r
import sys
import os
sys.path.append(os.path.abspath((os.path.join(os.getcwd(), '..'))))
from lbl8r.constants import *
from lbl8r.constants import XYLENA_PATH

# In[ ]:

root_path = Path("../")

data_path = root_path / XYLENA_PATH


out_path = data_path / "LBL8R"

# Reload the files and calculate the PCA of the expression values

# In[ ]:
train_filen = out_path / XYLENA_TRAIN.replace("_cnt.h5ad", "_exp_nb_out.h5ad")
exp_train_ad = ad.read_h5ad(train_filen)

# In[ ]:
# pcs
sc.pp.pca(exp_train_ad)

# In[ ]:
test_filen = out_path / XYLENA_TEST.replace("_cnt.h5ad", "_exp_nb_out.h5ad")
exp_test_ad = ad.read_h5ad(test_filen)
exp_test_ad = transfer_pcs(exp_train_ad, exp_test_ad)

# In[ ]:
# resave the adatas with the pcs
export_ouput_adata(exp_train_ad, XYLENA_TRAIN.replace("_cnt.h5ad", "_exp_nb.h5ad"), out_path)
export_ouput_adata(exp_test_ad, XYLENA_TEST.replace("_cnt.h5ad", "_exp_nb.h5ad"), out_path)


