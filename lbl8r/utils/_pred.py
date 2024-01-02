import torch.nn.functional as F
from torch.distributions import Categorical
import torch
import pandas as pd
import numpy as np
from anndata import AnnData
from scvi.model import SCVI
from scanpy.pp import pca

def calculate_margin_of_probability(probabilities):

    # Get the top two probabilities
    top_probs, _ = torch.topk(probabilities, 2)

    # Calculate the margin
    margin = top_probs[:,0] - top_probs[:,1]
    return margin.numpy()


def get_stats_from_logits(logits:torch.Tensor, categories:np.ndarray) -> dict:

    # Applying softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)

    # Create a Categorical distribution
    distribution = Categorical(probs=probabilities)

    # Calculate entropy
    entropy = distribution.entropy()

    # n_classes = probabilities.shape[1]

    logs = logits.numpy()
    probs = probabilities.numpy()
    ents = entropy.numpy()
    logents = entropy.log().numpy()

    # print("Logits: ", logs)
    # print("Probabilities: ", probs)

    maxprobs = probs.max(axis=1)

    labels = categories[probs.argmax(axis=1)]

 
    margin = calculate_margin_of_probability(probabilities)


    return {"logit":logs,
     "prob":probs,
     "entropy":ents,
     "logE":logents,
     "max_p":maxprobs,
     "mop":margin,
     "label":labels}



def get_stats_table(probabilities:dict, categories:np.ndarray, index:np.ndarray, soft:bool=True) -> pd.DataFrame:
        

    if not soft:
        preds_df = pd.DataFrame(probabilities['logit'], columns=categories, index=index)
        preds_df = preds_df.idxmax(axis=1)
    else:
        preds_df = pd.DataFrame(probabilities['logit'], 
                                columns=[f"lg_{c}" for c in categories], 
                                index=index)
        for i,c in enumerate(categories):
            preds_df[f"p_{c}"] = probabilities['prob'][:,i]

        preds_df['label'] = probabilities['label']
        preds_df['max_p'] = probabilities['max_p']
        preds_df['entropy'] = probabilities['entropy']
        preds_df['logE'] = probabilities['logE']
        preds_df['mop'] = probabilities['mop']


    return preds_df


# def make_latent_adata(vae, **get_latent_args):
#     # TODO: make it accept adata as input
#     #   also probably take the qzv.log() of the latent variables for nicer targets...
#     return_dist = get_latent_args.pop("return_dist", True)

#     if return_dist:
#         qzm, qzv = vae.get_latent_representation(return_dist=return_dist,**get_latent_args)
#         latent_adata = ad.AnnData(np.concatenate([qzm,qzv], axis=1))
#         # latent_adata.obsm["X_latent_qzm"] = qzm
#         # latent_adata.obsm["X_latent_qzv"] = qzv
#         var_names = [f"zm_{i}" for i in range(qzm.shape[1])]+[f"zv_{i}" for i in range(qzv.shape[1])]
#     else:
#         latent_adata = ad.AnnData(vae.get_latent_representation(**get_latent_args))
#         var_names = [f"z_{i}" for i in range(latent_adata.shape[1])]


#     latent_adata.obs_names = vae.adata.obs_names.copy()
#     latent_adata.obs = vae.adata.obs.copy()
#     latent_adata.var_names = var_names
#     return latent_adata



def make_latent_adata(scvi_model : SCVI, 
                        adata : AnnData,
                        return_dist: bool = True):
    

    if return_dist:
        qzm, qzv = scvi_model.get_latent_representation(adata,give_mean=False,return_dist=return_dist)
        latent_adata = AnnData(np.concatenate([qzm,qzv], axis=1))
        var_names = [f"zm_{i}" for i in range(qzm.shape[1])]+[f"zv_{i}" for i in range(qzv.shape[1])]
    else:
        latent_adata = AnnData(scvi_model.get_latent_representation(adata,give_mean=True))
        var_names = [f"z_{i}" for i in range(latent_adata.shape[1])]

    latent_adata.obs_names = adata.obs_names.copy()
    latent_adata.obs = adata.obs.copy()
    latent_adata.var_names = var_names
    print(latent_adata.shape)
    return latent_adata


def make_scvi_normalized_adata(scvi_model : SCVI, 
                        adata : AnnData):
 
    labels_key = 'cell_type'
    scvi_model.setup_anndata(adata,labels_key=labels_key, batch_key=None)
    denoised = scvi_model.get_normalized_expression(adata, library_size=1e4, return_numpy=True,)

    norm_adata = AnnData(denoised)

    norm_adata.obs_names = adata.obs_names.copy()
    norm_adata.obs = adata.obs.copy()
    norm_adata.var_names = adata.var_names.copy()
    print(norm_adata.shape)
    return norm_adata





def make_pc_loading_adata( adata : AnnData):

    if "X_pca" in adata.obsm.keys():
        # already have the loadings...
        loading_adata = AnnData(adata.obsm['X_pca'])
    else:
        pcs = pca(adata.X)
        loading_adata = AnnData(pcs)

    var_names = [f"pc_{i}" for i in range(loading_adata.shape[1])]

    loading_adata.obs_names = adata.obs_names.copy()
    loading_adata.obs = adata.obs.copy()
    loading_adata.var_names = var_names
    print(loading_adata.shape)
    return loading_adata


def add_predictions_to_adata(adata, predictions, insert_key="pred", pred_key="label"):
    obs = adata.obs
    if insert_key in obs.columns:
        # replace if its already there
        obs.drop(columns=[insert_key], inplace=True)

    adata.obs = pd.merge(obs, predictions[pred_key], left_index=True, right_index=True, how='left').rename(columns={pred_key:insert_key})

    return adata