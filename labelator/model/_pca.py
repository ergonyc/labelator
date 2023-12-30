import inspect
import os
import torch
import pickle
import numpy as np

from anndata import AnnData, read
from copy import deepcopy
from typing import Optional, Union

from ..module._classifier import Classifier
from ..module._pca import PCAClassifier
from ..trainer import ClassifierTrainer

from ..base._base import BaseMixin, SurgeryMixin

# Here we need to make a module which trains on the raw data, transforms to loadings, and then categorizes

class PCA(BaseMixin, SurgeryMixin):
    """Model for scArches class. This class contains the implementation of Classifying a PCA loading.

       Parameters
       ----------
       adata: : `~anndata.AnnData`
            Annotated data matrix. Has to be count data for 'nb' and 'zinb' loss and normalized log transformed data
            for 'mse' loss.
       condition_key: String
            column name of conditions in `adata.obs` data frame.
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integer
            Bottleneck layer (z)  size.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`==0 no dropout will be applied.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
    """
    def __init__(
        self,
        adata: AnnData,
        condition_key: str = None,
        conditions: Optional[list] = None,
        hidden_layer_sizes: list = [256, 64],
        latent_dim: int = 10,
        dr_rate: float = 0.05,
        use_bn: bool = False,
        use_ln: bool = True,
    ):
        self.adata = adata

        self.condition_key_ = condition_key

        if conditions is None:
            if condition_key is not None:
                self.conditions_ = adata.obs[condition_key].unique().tolist()
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions

        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.latent_dim_ = latent_dim
        self.dr_rate_ = dr_rate
        self.use_bn_ = use_bn
        self.use_ln_ = use_ln

        self.input_dim_ = adata.n_vars



        self.model = PCAClassifier(
            n_input=self.input_dim_,
            n_labels=len(self.conditions__,
            n_hidden=self.hidden_layer_sizes_,
            dropout_rate=self.dr_rate_,
            use_batch_norm=self.use_bn_,
            use_layer_norm=self.use_ln_,
        )

        self.is_trained_ = False

        self.trainer = None

    def train(
        self,
        n_epochs: int = 400,
        lr: float = 1e-3,
        eps: float = 0.01,
        **kwargs
    ):
        """Train the model.

           Parameters
           ----------
           n_epochs
                Number of epochs for training the model.
           lr
                Learning rate for training the model.
           eps
                torch.optim.Adam eps parameter
           kwargs
                kwargs for the TrVAE trainer.
        """
        self.trainer = ClassifierTrainer(
            self.model,
            self.adata,
            condition_key=self.condition_key_,
            **kwargs)
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True

    @classmethod
    def _get_init_params_from_dict(cls, dct):
        init_params = {
            'condition_key': dct['condition_key_'],
            'conditions': dct['conditions_'],
            'hidden_layer_sizes': dct['hidden_layer_sizes_'],
            'latent_dim': dct['latent_dim_'],
            'dr_rate': dct['dr_rate_'],
            'use_mmd': dct['use_mmd_'],
            'mmd_on': dct['mmd_on_'],
            'mmd_boundary': dct['mmd_boundary_'],
            'recon_loss': dct['recon_loss_'],
            'beta': dct['beta_'],
            'use_bn': dct['use_bn_'],
            'use_ln': dct['use_ln_'],
        }

        return init_params

    @classmethod
    def _validate_adata(cls, adata, dct):
        if adata.n_vars != dct['input_dim_']:
            raise ValueError("Incorrect var dimension")

        adata_conditions = adata.obs[dct['condition_key_']].unique().tolist()
        if not set(adata_conditions).issubset(dct['conditions_']):
            raise ValueError("Incorrect conditions")
