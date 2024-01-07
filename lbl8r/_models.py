from __future__ import annotations

import io
import logging
import warnings
from collections.abc import Sequence
from contextlib import redirect_stdout

import anndata
import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.dataloaders import DataSplitter
from scvi.model import SCVI
from scvi.model._utils import get_max_epochs_heuristic
from scvi.model.base import BaseModelClass

# from scvi.module import Classifier
from scvi.module.base import auto_move_data  #
from scvi.train import ClassifierTrainingPlan, LoudEarlyStopping, TrainRunner
from scvi.utils import setup_anndata_dsp
from scvi.utils._docstrings import devices_dsp

# from typing import Optional


from .utils._pred import get_stats_from_logits, get_stats_table
from ._modules import Classifier

logger = logging.getLogger(__name__)

LABELS_KEY = "cell_type"


def _validate_scvi_model(scvi_model: SCVI, restrict_to_batch: str):
    if scvi_model.summary_stats.n_batch > 1 and restrict_to_batch is None:
        warnings.warn(
            "(from Solo) should only be trained on one lane of data using `restrict_to_batch`. Performance may suffer.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )


class scviLBL8R(BaseModelClass):
    """Cell type classification in scRNA-seq

    Tinitialize the model using the class method
    :meth:`~lbl8r.scviLBL8R.from_scvi_model`, which takes as
    input a pre-trained :class:`~scvi.model.SCVI` object.

    [Derived from scvi.external.SOLO]

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
        Object should contain latent representation of real cells and doublets as `adata.X`.
        Object should also be registered, using `.X` and `labels_key="_solo_doub_sim"`.
    **classifier_kwargs
        Keyword args for :class:`~scvi.module.Classifier`

    Examples
    --------

    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCVI.setup_anndata(adata)
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> lbl8r = scviLBL8R.from_scvi_model(vae)
    >>> lbl8r.train()
    >>> lbl8r.predict()

    Notes
    -----

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

        self._from_scvi_model = False
        self._model_summary_string = "scviLBL8R model"
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: SCVI,
        adata: AnnData | None = None,
        restrict_to_batch: str | None = None,
        **classifier_kwargs,
    ):
        """Instantiate a scviLBL8R model from an scvi model.

        Parameters
        ----------
        scvi_model
            Pre-trained :class:`~scvi.model.SCVI` model. The AnnData object used to
            initialize this model should have only been setup with count data, and
            optionally a `batch_key`. Extra categorical and continuous covariates are
            currenty unsupported.
        adata
            Optional AnnData to use that is compatible with `scvi_model`.
        restrict_to_batch
            NOT USED. DEPRICATE. Batch category to restrict the (based on SOLO model needs)
            P to if `scvi_model` was set up with a `batch_key`. This allows the model to be trained on the subset of cells
            belonging to `restrict_to_batch` when `scvi_model` was trained on multiple
            batches. If `None`, all cells are used.
        **classifier_kwargs
            Keyword args for :class:`~scvi.module.Classifier`

        Returns
        -------
        scviLBL8R model
        """
        _validate_scvi_model(scvi_model, restrict_to_batch=restrict_to_batch)
        orig_adata_manager = scvi_model.adata_manager
        orig_batch_key_registry = orig_adata_manager.get_state_registry(
            REGISTRY_KEYS.BATCH_KEY
        )
        orig_labels_key_registry = orig_adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        )
        orig_batch_key = orig_batch_key_registry.original_key
        orig_labels_key = orig_labels_key_registry.original_key

        if len(orig_adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY)) > 0:
            raise ValueError(
                "Initializing a model from SCVI with registered continuous "
                "covariates is currently unsupported."
            )
        if len(orig_adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)) > 0:
            raise ValueError(
                "Initializing a model from SCVI with registered categorical "
                "covariates is currently unsupported."
            )
        scvi_trained_with_batch = len(orig_batch_key_registry.categorical_mapping) > 1
        if not scvi_trained_with_batch and restrict_to_batch is not None:
            raise ValueError(
                "Cannot specify `restrict_to_batch` when initializing a model from SCVI "
                "not trained with multiple batches."
            )
        if scvi_trained_with_batch > 1 and restrict_to_batch is None:
            warnings.warn(
                "`restrict_to_batch` not specified but `scvi_model` was trained with "
                "multiple batches. simulated using the first batch.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        if adata is not None:
            adata_manager = orig_adata_manager.transfer_fields(adata)
            cls.register_manager(adata_manager)
        else:
            adata_manager = orig_adata_manager
        adata = adata_manager.adata

        # I should be able to remove this, since I'm not using the batch key
        if restrict_to_batch is not None:
            batch_mask = adata.obs[orig_batch_key] == restrict_to_batch
            if np.sum(batch_mask) == 0:
                raise ValueError(
                    "Batch category given to restrict_to_batch not found.\n"
                    + "Available categories: {}".format(
                        adata.obs[orig_batch_key].astype("category").cat.categories
                    )
                )
            # indices in adata with restrict_to_batch category
            batch_indices = np.where(batch_mask)[0]
        else:
            # use all indices
            batch_indices = None

        # if model is using observed lib size, needs to get lib sample
        # which is just observed lib size on log scale
        give_mean_lib = not scvi_model.module.use_observed_lib_size

        return_dist = classifier_kwargs.pop("return_dist", False)
        cls.return_dist = return_dist

        if return_dist:
            qzm, qzv = scvi_model.get_latent_representation(return_dist=return_dist)
            latent_adata = AnnData(np.concatenate([qzm, qzv], axis=1))
            # latent_adata.obsm["X_latent_qzm"] = qzm
            # latent_adata.obsm["X_latent_qzv"] = qzv
            var_names = [f"zm_{i}" for i in range(qzm.shape[1])] + [
                f"zv_{i}" for i in range(qzv.shape[1])
            ]
        else:
            latent_adata = AnnData(scvi_model.get_latent_representation())
            var_names = [f"z_{i}" for i in range(latent_adata.shape[1])]

        # latent_adata.obs_names = scvi_model.adata.obs_names.copy()
        # latent_adata.obs = scvi_model.adata.obs.copy()
        # latent_adata.var_names = var_names
        latent_adata.obs_names = adata.obs_names.copy()
        latent_adata.obs = adata.obs.copy()
        latent_adata.var_names = var_names
        latent_adata.obsm = adata.obsm.copy()
        latent_adata.uns = {}

        # latent_adata = cls.get_latent_adata(scvi_model,adata,return_dist=return_dist)
        cls.setup_anndata(latent_adata, labels_key=LABELS_KEY)
        cls._from_scvi_model = True
        return cls(latent_adata, **classifier_kwargs)

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
        """Return classification predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`scviLBL8R.setup_anndata`.
        probs
            returns dictionary of stats instead of dataframe. makes soft moot
        soft
            Return probabilities instead of class label


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
            printstr = (
                f"adata from .from_scvi_model"
                if self._from_scvi_model
                else f"loading adata"
            )
            print(f"{printstr} size={adata.shape}")

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

        cats = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        ).categorical_mapping

        stats = get_stats_from_logits(y_pred, cats)

        if probs:
            return stats

        return get_stats_table(
            probabilities=stats, categories=cats, index=adata.obs_names, soft=soft
        )

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

        # if cls.return_dist:
        #     qzm, qzv = scvi_model.get_latent_representation(return_dist=cls.return_dist)
        #     latent_adata = AnnData(np.concatenate([qzm,qzv], axis=1))
        #     # latent_adata.obsm["X_latent_qzm"] = qzm
        #     # latent_adata.obsm["X_latent_qzv"] = qzv
        #     var_names = [f"zm_{i}" for i in range(qzm.shape[1])]+[f"zv_{i}" for i in range(qzv.shape[1])]
        # else:
        #     latent_adata = AnnData(scvi_model.get_latent_representation())
        #     var_names = [f"z_{i}" for i in range(latent_adata.shape[1])]

        # latent_adata.obs_names = scvi_model.adata.obs_names.copy()
        # latent_adata.obs = scvi_model.adata.obs.copy()
        # latent_adata.var_names = var_names
        # # latent_adata = cls.get_latent_adata(scvi_model,adata,return_dist=return_dist)
        # cls.setup_anndata(latent_adata, labels_key=LABELS_KEY)
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


# TODO:  add from_embedding to LBL8R to automatically create the input_type
class LBL8R(BaseModelClass):
    """.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.LinearSCVI.setup_anndata`.
    n_labels
        Number of labels to clasify.
    input_type
        data type of input to add to model summary string. None is default "LBL8R model"
    **classifier_kwargs
        Keyword args for :class:`Classifier`
    """

    def __init__(
        self,
        adata: AnnData,
        n_labels: int = 8,
        input_type: str | None = None,
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

        self._model_summary_string = (
            "LBL8R model" if input_type is None else f"{input_type} LBL8R model"
        )
        self.init_params_ = self._get_init_params(locals())

    # def get_loadings(self) -> pd.DataFrame:
    #     """Extract per-gene weights in the linear decoder.

    #     Shape is genes by `n_latent`. UNTESTED

    #     """
    #     cols = [f"Z_{i}" for i in range(self.n_latent)]
    #     var_names = self.adata.var_names
    #     loadings = pd.DataFrame(
    #         self.module.get_loadings(), index=var_names, columns=cols
    #     )

    #     return loadings

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

    @torch.inference_mode()
    def predict(
        self,
        adata: AnnData | None = None,
        probs: bool = True,
        soft: bool = True,
    ) -> pd.DataFrame | dict:
        """Return classification predictions.

        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`LBL8R.setup_anndata`.
        probs
            returns dictionary of stats instead of dataframe. makes soft moot
        soft
            Return probabilities instead of class label


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

        cats = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        ).categorical_mapping

        stats = get_stats_from_logits(y_pred, cats)

        if probs:
            return stats

        return get_stats_table(
            probabilities=stats, categories=cats, index=adata.obs_names, soft=soft
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


# # this is just a generic classifier witn an AnnData input.  Shouldn't be named pca, since
# #        attempts to wrap the PCA embedding into the model were incomplete
# #  simply renaming to LBL8R _should_ work, but _model_summary_string etc might cause problems... s
# #  so i'm just going to copy the whole thing and rename it
# class pcaLBL8R(BaseModelClass):
#     """.

#     Parameters
#     ----------
#     adata
#         AnnData object that has been registered via :meth:`~scvi.model.LinearSCVI.setup_anndata`.
#     n_hidden
#         Number of nodes per hidden layer.
#     n_latent
#         Dimensionality of the latent space.
#     n_layers
#         Number of hidden layers used for encoder NN.
#     dropout_rate
#         Dropout rate for neural networks.

#     """

#     def __init__(
#         self,
#         adata: AnnData,
#         n_labels: int = 8,
#         input_type: str = "pca",
#         **classifier_kwargs,
#     ):
#         # TODO, catch user warning here and logger warning
#         # about non count data
#         super().__init__(adata)

#         self.n_labels = n_labels
#         self.module = Classifier(
#             n_input=self.summary_stats.n_vars,
#             n_labels=self.n_labels,
#             logits=True,
#             **classifier_kwargs,
#         )

#         self._model_summary_string = "pcaLBL8R model"
#         self.init_params_ = self._get_init_params(locals())

#     def get_loadings(self) -> pd.DataFrame:
#         """Extract per-gene weights in the linear decoder.

#         Shape is genes by `n_latent`. UNTESTED

#         """
#         cols = [f"Z_{i}" for i in range(self.n_latent)]
#         var_names = self.adata.var_names
#         loadings = pd.DataFrame(
#             self.module.get_loadings(), index=var_names, columns=cols
#         )

#         return loadings

#     @classmethod
#     @setup_anndata_dsp.dedent
#     def setup_anndata(
#         cls,
#         adata: AnnData,
#         labels_key: str | None = None,
#         layer: str | None = None,
#         **kwargs,
#     ):
#         """%(summary)s.

#         Parameters
#         ----------
#         %(param_labels_key)s
#         %(param_layer)s
#         """
#         setup_method_args = cls._get_setup_method_args(**locals())
#         anndata_fields = [
#             LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
#             CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
#         ]
#         adata_manager = AnnDataManager(
#             fields=anndata_fields, setup_method_args=setup_method_args
#         )
#         adata_manager.register_fields(adata, **kwargs)
#         cls.register_manager(adata_manager)

#     @torch.inference_mode()
#     def predict(
#         self,
#         adata: AnnData | None = None,
#         probs: bool = True,
#         soft: bool = True,
#     ) -> pd.DataFrame | dict:
#         """Return classification predictions.

#         Parameters
#         ----------
#         adata
#             AnnData object that has been registered via :meth:`pcaLBL8R.setup_anndata`.
#         probs
#             returns dictionary of stats instead of dataframe. makes soft moot
#         soft
#             Return probabilities instead of class label


#         Returns
#         -------
#         DataFrame with prediction, index corresponding to cell barcode.
#         """
#         # adata = self._validate_anndata(None)

#         if adata is None:
#             adata = self._validate_anndata(None)
#         else:
#             # not sure if i should pass scvi_model or get it from the registry??
#             # adata = self.get_latent_adata(adata,scvi_model)
#             adata = self._validate_anndata(adata)
#             print(adata.shape)

#         scdl = self._make_data_loader(
#             adata=adata,
#         )

#         @auto_move_data
#         def auto_forward(module, x):
#             return module(x)

#         y_pred = []
#         for _, tensors in enumerate(scdl):
#             x = tensors[REGISTRY_KEYS.X_KEY]
#             pred = auto_forward(self.module, x)
#             y_pred.append(pred.cpu())

#         y_pred = torch.cat(y_pred)

#         cats = self.adata_manager.get_state_registry(
#             REGISTRY_KEYS.LABELS_KEY
#         ).categorical_mapping

#         stats = get_stats_from_logits(y_pred, cats)

#         if probs:
#             return stats

#         return get_stats_table(
#             probabilities=stats, categories=cats, index=adata.obs_names, soft=soft
#         )

#     @devices_dsp.dedent
#     def train(
#         self,
#         max_epochs: int = 400,
#         lr: float = 1e-3,
#         accelerator: str = "auto",
#         devices: int | list[int] | str = "auto",
#         train_size: float = 0.9,
#         validation_size: float | None = None,
#         shuffle_set_split: bool = True,
#         batch_size: int = 128,
#         datasplitter_kwargs: dict | None = None,
#         plan_kwargs: dict | None = None,
#         early_stopping: bool = True,
#         early_stopping_patience: int = 30,
#         early_stopping_min_delta: float = 0.0,
#         **kwargs,
#     ):
#         """Trains the model.

#         Parameters
#         ----------
#         max_epochs
#             Number of epochs to train for
#         lr
#             Learning rate for optimization.
#         %(param_accelerator)s
#         %(param_devices)s
#         train_size
#             Size of training set in the range [0.0, 1.0].
#         validation_size
#             Size of the test set. If `None`, defaults to 1 - `train_size`. If
#             `train_size + validation_size < 1`, the remaining cells belong to a test set.
#         shuffle_set_split
#             Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
#             sequential order of the data according to `validation_size` and `train_size` percentages.
#         batch_size
#             Minibatch size to use during training.
#         datasplitter_kwargs
#             Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
#         plan_kwargs
#             Keyword args for :class:`~scvi.train.ClassifierTrainingPlan`. Keyword arguments passed to
#         early_stopping
#             Adds callback for early stopping on validation_loss
#         early_stopping_patience
#             Number of times early stopping metric can not improve over early_stopping_min_delta
#         early_stopping_min_delta
#             Threshold for counting an epoch torwards patience
#             `train()` will overwrite values present in `plan_kwargs`, when appropriate.
#         **kwargs
#             Other keyword args for :class:`~scvi.train.Trainer`.
#         """
#         update_dict = {
#             "lr": lr,
#         }
#         if plan_kwargs is not None:
#             plan_kwargs.update(update_dict)
#         else:
#             plan_kwargs = update_dict

#         datasplitter_kwargs = datasplitter_kwargs or {}

#         if early_stopping:
#             early_stopping_callback = [
#                 LoudEarlyStopping(
#                     monitor="validation_loss" if train_size != 1.0 else "train_loss",
#                     min_delta=early_stopping_min_delta,
#                     patience=early_stopping_patience,
#                     mode="min",
#                 )
#             ]
#             if "callbacks" in kwargs:
#                 kwargs["callbacks"] += early_stopping_callback
#             else:
#                 kwargs["callbacks"] = early_stopping_callback
#             kwargs["check_val_every_n_epoch"] = 1

#         if max_epochs is None:
#             max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

#         plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

#         data_splitter = DataSplitter(
#             self.adata_manager,
#             train_size=train_size,
#             validation_size=validation_size,
#             shuffle_set_split=shuffle_set_split,
#             batch_size=batch_size,
#             **datasplitter_kwargs,
#         )
#         training_plan = ClassifierTrainingPlan(self.module, **plan_kwargs)
#         runner = TrainRunner(
#             self,
#             training_plan=training_plan,
#             data_splitter=data_splitter,
#             max_epochs=max_epochs,
#             accelerator=accelerator,
#             devices=devices,
#             **kwargs,
#         )
#         return runner()
