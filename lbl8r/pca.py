import logging
from typing import Literal, Optional
from torch import nn

import pandas as pd
from anndata import AnnData

from scvi.module import Classifier
from scvi.module.base import (
    BaseModuleClass,
    LossOutput
    auto_move_data,
)
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import DataSplitter
from scvi.utils import setup_anndata_dsp
from scvi.model.base import BaseModelClass
from scvi.train import ClassifierTrainingPlan, LoudEarlyStopping, TrainRunner

from .utils._pred import get_stats_from_logits, get_stats_table

logger = logging.getLogger(__name__)

LABELS_KEY = "cell_type"

class PCALabelator(BaseModelClass):
    """.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.LinearSCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder NN.
    dropout_rate
        Dropout rate for neural networks.

    """

    def __init__(
        self,
        adata: AnnData,
        n_labels: int = 8,
        **classifier_kwargs,
    ):
        # TODO, catch user warning here and logger warning
        # about non count data
        super().__init__(adata)

        self.n_labels = n_labels
        self.module = Classifier(
            n_input=self.summary_stats.n_vars,
            n_labels=self.n_labels,
            logits=True,
            **classifier_kwargs,
        )
    
        self._model_summary_string = "pcaLBL8R model"
        self.init_params_ = self._get_init_params(locals())


    def get_loadings(self) -> pd.DataFrame:
        """Extract per-gene weights in the linear decoder.

        Shape is genes by `n_latent`.

        """
        cols = [f"Z_{i}" for i in range(self.n_latent)]
        var_names = self.adata.var_names
        loadings = pd.DataFrame(
            self.module.get_loadings(), index=var_names, columns=cols
        )

        return loadings

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str | None = None,
        layer: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_labels_key)s
        %(param_layer)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)



class PCA(BaseModuleClass):
    """
    Skeleton model.

    Parameters
    ----------
    n_input
        Number of input genes
    n_latent
        Dimensionality of the latent space
    """

    def __init__(
        self,
        pcs: np.ndarray,
        n_labels: int = 8,
    ):
        super().__init__()

        self.pcs = pcs
        self.n_input = self.pcs.shape[1]
        self.decoder = Classifier(
            n_input=n_input,
            n_labels=self.n_labels,
            logits=True,
            **classifier_kwargs,
        )



    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 400,
        lr: float = 1e-3,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 30,
        early_stopping_min_delta: float = 0.0,
        **kwargs,
    ):
        """Trains the model.

        Parameters
        ----------
        max_epochs
            Number of epochs to train for
        lr
            Learning rate for optimization.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
            sequential order of the data according to `validation_size` and `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.ClassifierTrainingPlan`. Keyword arguments passed to
        early_stopping
            Adds callback for early stopping on validation_loss
        early_stopping_patience
            Number of times early stopping metric can not improve over early_stopping_min_delta
        early_stopping_min_delta
            Threshold for counting an epoch torwards patience
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        update_dict = {
            "lr": lr,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        datasplitter_kwargs = datasplitter_kwargs or {}

        if early_stopping:
            early_stopping_callback = [
                LoudEarlyStopping(
                    monitor="validation_loss" if train_size != 1.0 else "train_loss",
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    mode="min",
                )
            ]
            if "callbacks" in kwargs:
                kwargs["callbacks"] += early_stopping_callback
            else:
                kwargs["callbacks"] = early_stopping_callback
            kwargs["check_val_every_n_epoch"] = 1

        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            **datasplitter_kwargs,
        )
        training_plan = ClassifierTrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **kwargs,
        )
        return runner()

    @torch.inference_mode()
    def predict(
        self, 
        adata: AnnData | None = None,
        probs: bool = True,
        soft: bool = True, 
    ) -> pd.DataFrame | dict:
        """Return doublet predictions.

        Parameters
        ----------
        soft
            Return probabilities instead of class label
        include_simulated_doublets
            Return probabilities for simulated doublets as well.

        Returns
        -------
        DataFrame with prediction, index corresponding to cell barcode.
        """
        # adata = self._validate_anndata(None)

        if adata is None:
           adata = self._validate_anndata(None)
        else:
            # not sure if i should pass scvi_model or get it from the registry??
            # adata = self.get_latent_adata(adata,scvi_model)
            adata = self._validate_anndata(adata)
            print(adata.shape)

        scdl = self._make_data_loader(
            adata=adata,
        )

        @auto_move_data
        def auto_forward(module, x):
            return module(x)

        y_pred = []
        for _, tensors in enumerate(scdl):
            x = tensors[REGISTRY_KEYS.X_KEY]
            pred = auto_forward(self.module, x)
            y_pred.append(pred.cpu())

        y_pred = torch.cat(y_pred)

        cats = self.adata_manager.get_state_registry( REGISTRY_KEYS.LABELS_KEY ).categorical_mapping
    
        stats = get_stats_from_logits(y_pred,cats)
        
        if probs: return stats

        return get_stats_table(probabilities=stats, categories=cats, index=adata.obs_names, soft=soft)

