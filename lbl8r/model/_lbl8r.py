from __future__ import annotations

import logging
import warnings

import torch.nn.functional as F
from torch.distributions import Categorical
import torch
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pathlib import Path

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

from .module._classifier import Classifier
from ._scvi import get_trained_scvi

from .utils._pred import get_stats_table
from .utils._data import make_pc_loading_adata, add_pc_loadings
from .utils._timing import Timing
from .utils._artifact import model_exists
from .utils._pca import compute_pcs

from .._constants import PCA_KEY

# TODO: enable logging
# logger = logging.getLogger(__name__)

LABELS_KEY = "cell_type"


def prep_pcs_adata(
    adata: AnnData,
    pcs: np.ndarray | None = None,
    pca_key: str = "X_pca",
) -> AnnData:
    """
    make an adata with PCs copied to adata.X.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    pca_key : str
        Key for pca loadings. Default is `X_pca`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    if pcs is None:
        pcs = compute_pcs(adata)

    loadings_ad = make_pc_loading_adata(adata, pcs=pcs, pca_key=pca_key)
    return loadings_ad


def prep_raw_adata(
    adata: AnnData,
    pcs: np.ndarray | None = None,
    pca_key: str = "X_pca",
) -> AnnData:
    """
    make an adata with PCs copied to adata.X.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    pca_key : str
        Key for pca loadings. Default is `X_pca`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    if pcs is None:
        pcs = compute_pcs(adata)

    adata = add_pc_loadings(adata, pcs=pcs, pca_key=pca_key)
    return adata


def get_stats_from_logits(logits: torch.Tensor, categories: np.ndarray) -> dict:
    """
    Get probabilities, entropy, log entropy, labels, and margin of probability from logits.

    Parameters
    ----------
    logits : torch.Tensor
        Logits from a model.
    categories : np.ndarray
        Array of categories.

    Returns
    -------
    dict
        Dictionary of probabilities, entropy, log entropy, labels, and margin of probability.

    """
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

    return {
        "logit": logs,
        "prob": probs,
        "entropy": ents,
        "logE": logents,
        "max_p": maxprobs,
        "mop": margin,
        "label": labels,
    }


def calculate_margin_of_probability(probabilities: torch.Tensor) -> np.ndarray:
    """
    Calculate the margin of probability.

    Parameters
    ----------
    probabilities : torch.Tensor
        Probabilities from a model.

    Returns
    -------
    np.ndarray
        Array of margin of probability.
    """
    # Get the top two probabilities
    top_probs, _ = torch.topk(probabilities, 2)

    # Calculate the margin
    margin = top_probs[:, 0] - top_probs[:, 1]
    return margin.numpy()


def _validate_scvi_model(scvi_model: SCVI, restrict_to_batch: str):
    if scvi_model.summary_stats.n_batch > 1 and restrict_to_batch is None:
        warnings.warn(
            "(from Solo) should only be trained on one lane of data using `restrict_to_batch`. Performance may suffer.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )


# depricated
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


#
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

        # TODO: make this module either an XGB or Classifier...  so we are just changing the classifier type...
        #  will require making a shim for loading data so low priority
        self.module = Classifier(
            n_input=self.summary_stats.n_vars,
            n_labels=self.n_labels,
            logits=True,
            **classifier_kwargs,
        )

        self._model_summary_string = (
            "LBL8R classifier model"
            if input_type is None
            else f"{input_type} LBL8R classifier model"
        )
        self.init_params_ = self._get_init_params(locals())

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
        return runner()  # execute the runner and return results


@Timing(prefix="model_name")
def get_lbl8r(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "lbl8r",
    **training_kwargs,
) -> tuple[LBL8R, AnnData]:
    """
    Get the LBL8R model for single-cell data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    model_name : str
        Name of the model. Default is `lbl8r`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    **training_kwargs : dict
        Additional arguments to pass to `scvi.model.SCVI.train`.

    Returns
    -------
    LBL8R
        LBL8R model.
    AnnData
        Annotated data matrix with latent variables.
    """
    lbl8r_path = model_path / model_name
    labels_key = labels_key
    n_labels = len(adata.obs[labels_key].cat.categories)

    lbl8r_epochs = 200
    batch_size = 512

    # TODO: test. not sure I need this step
    LBL8R.setup_anndata(adata, labels_key=labels_key)

    # 1. load/train model
    if model_exists(lbl8r_path) and not retrain:
        print(f"`get_lbl8r` existing model from {lbl8r_path}")
        lat_lbl8r = LBL8R.load(lbl8r_path, adata.copy())

    else:
        lat_lbl8r = LBL8R(adata, n_labels=n_labels)
        lat_lbl8r.train(
            max_epochs=lbl8r_epochs,
            train_size=0.85,
            batch_size=batch_size,
            early_stopping=True,
            **training_kwargs,
        )

    if retrain or not model_exists(lbl8r_path):
        # save the reference model
        lat_lbl8r.save(lbl8r_path, overwrite=True)

    return lat_lbl8r, adata


def query_lbl8r_raw(
    adata: AnnData,
    labelator: LBL8R,
) -> pd.DataFrame:
    # ) -> AnnData:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labelator : scviLBL8R, pcaLBL8R, etc
        An classification model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    # labelator.setup_anndata(adata, labels_key=labels_key)  # "dummy")

    predictions = labelator.predict(adata, probs=False, soft=True)
    # loadings_ad = add_predictions_to_adata(
    #     adata, predictions, insert_key=INSERT_KEY, pred_key=PRED_KEY
    # )
    # adata = merge_into_obs(adata, predictions)
    # return adata
    return predictions


# depricated...
def get_scvi_lbl8r(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "scvi_nobatch",
    **training_kwargs,
):
    """
    Get scVI model and latent representation for `LBL8R` model. Note that `batch_key=None`
    Just a wrapper for `get_trained_scvi` with `batch_key=None`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCVI_nobatch`.
    **training_kwargs : dict
        Additional arguments to pass to `scvi.model.SCVI.train`.

    Returns
    -------
    SCVI
        scVI model.
    AnnData
        Annotated data matrix with latent variables.

    """

    # 0 . load or train an scvi vae
    # just call get_trained_scvi with batch_key=None
    vae, adata = get_trained_scvi(
        adata,
        labels_key=labels_key,
        batch_key=None,
        model_path=model_path,
        retrain=retrain,
        model_name=model_name,
        **training_kwargs,
    )

    # 1. get the latent representation
    latent_ad = prep_pcs_adata(adata, vae=vae, labels_key=labels_key)
    # 2. get the lbl8r classifier
    vae_lbl8r, latent_ad = get_lbl8r(
        latent_ad,
        labels_key=labels_key,
        model_path=model_path,
        retrain=retrain,
        model_name=model_name,
        **training_kwargs,
    )

    return vae_lbl8r, vae, latent_ad


# depricated...
def get_pca_lbl8r(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "LBL8R_pca",
    **training_kwargs,
):
    """
    just a wrapper for get_lbl8r that defaults to modelname = LBL8R_pca
    """
    #

    # 1. get the latent representation
    pcs_ad = prep_pcs_adata(adata, pca_key=PCA_KEY)
    # 2. get the lbl8r classifier
    pcs_lbl8r, pcs_ad = get_lbl8r(
        pcs_ad,
        labels_key=labels_key,
        model_path=model_path,
        retrain=retrain,
        model_name=model_name,
        **training_kwargs,
    )

    return pcs_lbl8r, pcs_ad
