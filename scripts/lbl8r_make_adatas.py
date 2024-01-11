#!/usr/bin/env python
# coding: utf-8

# ## LBL8R create adata embeddings
# 
# ### overview.
# This notebook is a simplified version of `lbl8r_scvi.ipynb` which will not train any of the model but will load and prep `anndata` ("annotated data") files to be used downstream by the `LBL8R`.
# 
# 
# 
# ### Models and Embeddings
# 
# We will use a variety of models to "embed" the scRNAseq counts into lower dimension.
# - scVI latents
# - PCA. We interpret this as a linear embedding
# - etc.  in the future non-variational Auto Encoders, or other "compressions" 
# 
# ### files
# We will make 5 sets of files from Xylena's dataset from both the "test" and "train" subsets:
# - raw counts (0)
#     - PCA embedding (1.)
#     - scVI embeddings 
#         - mean latent only (2. )
#         - mean and var latents (concatenated) (3. )
# - normalized expression (scVI)
#     - normalized expression @ 1e4 `library_size`(4. )
#     - PCA embeddings of above (5. )

# In[1]:


import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    get_ipython().system('pip uninstall -y typing_extensions')
    get_ipython().system('pip install --quiet scvi-colab')
    from scvi_colab import install
    install()

else:
    import os
    # os.chdir('../')

    ### import local python functions in ../lbl8r
    sys.path.append(os.path.abspath((os.path.join(os.getcwd(), '..'))))


# In[2]:


from pathlib import Path
from scvi.model import SCVI

import scanpy as sc

import numpy as np
import anndata as ad


if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    
from lbl8r.utils import mde, make_latent_adata, make_scvi_normalized_adata, make_pc_loading_adata


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Load Train, Validate Data 

# In[3]:


if IN_COLAB:
    root_path = Path("/content/drive/MyDrive/")
    data_path = root_path / "SingleCellModel/data"
else:
    root_path = Path("../")
    if sys.platform == "darwin":
        data_path = root_path / "data/xylena_raw"
    else:
        data_path = root_path / "data/scdata/xylena_raw"
        
XYLENA_ANNDATA = "brain_atlas_anndata.h5ad"
XYLENA_METADATA = "final_metadata.csv"
XYLENA_ANNDATA2 = "brain_atlas_anndata_updated.h5ad"

XYLENA_TRAIN = XYLENA_ANNDATA.replace(".h5ad", "_train.h5ad")
XYLENA_TEST = XYLENA_ANNDATA.replace(".h5ad", "_test.h5ad")


cell_type_key = 'cell_type'


# ## load scVI model 

# 

# In[4]:


model_path = root_path / "lbl8r_models"
scvi_path = model_path / "scvi_nobatch"

labels_key = 'cell_type'



# ### setup train data for scVI

# In[5]:


outfilen = data_path / XYLENA_TRAIN
train_ad = ad.read_h5ad(outfilen)


# In[6]:


train_ad.obs.cell_type.cat.categories


# In[7]:


train_ad.obs[['seurat_clusters','cell_type','type','tmp']][train_ad.obs.cell_type == "Unknown"]


# In[8]:


train_ad.obs[['seurat_clusters','cell_type','type','tmp']].drop_duplicates()



# In[9]:


SCVI.setup_anndata(train_ad,labels_key=labels_key, batch_key=None) #"dummy")


# ### load trained scVI

# In[10]:


vae = SCVI.load(scvi_path.as_posix(),train_ad.copy())


# --------------
# ## make scVI normalized adata for further testing... i.e. `pcaLBL8R`

# In[11]:


norm_train_ad = make_scvi_normalized_adata(vae, train_ad)
norm_train_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_train_scvi_normalized.h5ad"))


# ## Now make on the latent anndata

# In[12]:


scvi_train_ad = make_latent_adata(vae,train_ad, return_dist=True)
scvi_train_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_train_scVI_lat.h5ad"))
del scvi_train_ad


# In[13]:


scvi_train_ad_mu = make_latent_adata(vae,train_ad, return_dist=False)
scvi_train_ad_mu.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_train_scVImu_lat.h5ad"))
del scvi_train_ad_mu


# ## PCA `AnnData` files

# In[14]:


loadings_train_ad = make_pc_loading_adata( train_ad)
loadings_train_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_train_pca.h5ad"))


# In[15]:


norm_loadings_train_ad = make_pc_loading_adata( norm_train_ad)
norm_loadings_train_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_train_scvi_normalized_pca.h5ad"))


# In[16]:


del train_ad
del norm_train_ad


# ------------------
# Now test data
# 
# 1. setup anndata
# 2. get scVI normalized expression
# 3. get scVI latents
# 
# 

# In[17]:


filen = data_path / XYLENA_TEST
test_ad = ad.read_h5ad(filen)


# In[18]:


test_ad


# In[19]:


SCVI.setup_anndata(test_ad.copy(),labels_key=labels_key, batch_key=None) #"dummy")


# In[20]:


norm_test_ad = make_scvi_normalized_adata(vae, test_ad)
norm_test_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_test_scvi_normalized.h5ad"))



# In[21]:


scVIqzmd_test_ad = make_latent_adata(vae,test_ad, return_dist=True)
scVIqzmd_test_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_test_scVI_qzmv.h5ad"))

del scVIqzmd_test_ad



# In[22]:


scVIz_test_ad = make_latent_adata(vae, test_ad, return_dist=False)
scVIz_test_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_test_scVI_z.h5ad"))

del scVIz_test_ad


# ## PCA `AnnData` files

# In[23]:


loadings_test_ad = make_pc_loading_adata( test_ad)
loadings_test_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_test_pca.h5ad"))


# In[24]:


norm_loadings_test_ad = make_pc_loading_adata( norm_test_ad)
norm_loadings_test_ad.write_h5ad(data_path / XYLENA_ANNDATA.replace(".h5ad", "_test_scvi_normalized_pca.h5ad"))


# In[ ]:





# In[ ]:





# In[ ]:




