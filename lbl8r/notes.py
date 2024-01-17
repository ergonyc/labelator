## wrappers for getting trained models

from lbl8r import (
    get_lbl8r,
    get_lbl8r,
    query_lbl8r,
    get_scvi_lbl8r,
    get_trained_scvi,
    get_trained_scanvi,
)
from lbl8r._constants import *
from lbl8r._constants import XYLENA_PATH


def get_model(
    adata,
    model_name,
    mode,
    model_path,
    label_key,
    retrain=False,
    plot_training=False,
    **fig_kwargs,
):
    """
    Get a model.
    """
    if model_name == "scvi":
        model = get_trained_scvi(adata, mode, model_path)

        vae_model_name = "scvi"
        (
            vae,
            train_ad,
        ) = get_scvi_lbl8r(  # sa,e ast get_trained_scvi but forces "batch"=None
            train_ad,
            labels_key=label_key,
            model_path=model_path,
            retrain=retrain,
            model_name=vae_model_name,
            plot_training=plot_training,
            **fig_kwargs,
        )

    elif model_name == "lbl8r":
        model = get_lbl8r(adata, mode, model_path)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model


# ------------

# mac_load / mac_get random code
"""
strategy:
- create all the paths for models and artifacts at setup.

0. create the paths, set the args for the pipeline

Pipeline:
1. load the "input" data
2. get the trained model
3. query the model
4. create the artifacts
    - figures
    - models
    - auxiliary data
        - latent versions
        - pcs version
        - scvi normalized version
    - predictions updated metadata (tabular)
    - vars (genes)



Pipeline 2:  (query pre-existing model)
- scanvi model:
    - load the model
        - create auxilarry data:
    - create/fit surgery model
    - query the surgery model
    - create artifacts:
        - figures
        - models
        - auxiliary data
            - latent versions
            - pcs version
            - scvi normalized version
        - predictions updated metadata
        - vars (genes) list of genes used for training, maybe include variance?
        

artifacts:
- figures
- predcitions (tables)
- latents (adata)

TODO: 
    1. simplify "dataprep" routines. i.e. don't add all the metadat, just put it somewhere
    2. make stubs for .var diversity.  i.e. reference to the 3k genes being trained on
        - how to deal with "empty" genes in query data?
    3. simplify artifacts to save PCs, naming convention for table artifacts (obs) of predictions 
        - parquet vs. csv?
    4. cli (click) interface
    5 helpers to re-aggregate tabular data with adata for easy visulaization


"""


from anndata import AnnData
from scvi.model import SCVI, SCANVI
from pathlib import Path

from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from .model._lbl8r import LBL8R
from .utils import (
    plot_lbl8r_training,
)

from ._constants import *
from .model._xgb import train_xgboost, load_xgboost, get_xgb_data
from ._artifacts import Adata, Model

PRED_KEY = "label"
INSERT_KEY = "pred"
# Model = SCVI | SCANVI | LBL8R | Booster


def get_model(
    adata: Adata,
    model_name: str,
    model_path: Path | str,
    model: SCVI | SCANVI | None = None,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    retrain: bool = False,
    **training_kwargs,
) -> (Model, Adata):
    """
    Load a model from the model_path

    Parameters
    ----------
    model_name : str
        Name of the model.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.

    Returns
    -------
    SCVI | SCANVI | LBL8R | Booster
        A model object.

    """

    model_path = model_path / model_name
    if not model_path.exists():
        raise ValueError(f"{model_path} does not exist")

    ad = adata.adata

    if model_name.startswith("scvi"):
        model = get_trained_scvi(
            adata.adata,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("scanvi"):
        if model is None:
            ValueError("SCANVI model must be passed")

        model = get_trained_scanvi(
            adata.adata,
            vae=model,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("lbl8r"):
        model = get_lbl8r(
            ad,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("xgb"):
        model = get_xgb(
            ad,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("query_scvi"):
        if model is None:
            ValueError("SCANVI model must be passed")

        model = get_query_scvi(
            ad,
            vae=model,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    elif model_name.startswith("query_scanvi"):
        if model is None:
            ValueError("SCANVI model must be passed")

        model = get_query_scanvi(
            ad,
            scanvi_model=model,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )

    else:
        raise ValueError(f"unknown model_name {model_name}")

    return model


# TODO: add save and load flags so we can use the functions and NOT overwrite on accident
def get_trained_scvi(
    adata: AnnData,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "scvi",
    **training_kwargs,
) -> (SCVI, AnnData):
    """
    Get scVI model and add latent representation to adata

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    batch_key : str
        Key for batch labels. Default is `None`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCVI`.

    Returns
    -------
    SCVI
        scVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    scvi_epochs = 200
    batch_size = 512

    layer = None
    categorical_covariate_keys = (
        None  # ['sample', 'batch'] Currently limited to single categorical...
    )
    continuous_covariate_keys = (
        None  # noise = ['doublet_score', 'percent.mt', 'percent.rb'] # aka "noise"
    )
    size_factor_key = None  # library size

    SCVI.setup_anndata(
        adata,
        batch_key=batch_key,  # using this to prevent issues with categorical covariates
        layer=layer,
        labels_key=labels_key,
        categorical_covariate_keys=categorical_covariate_keys,
        continuous_covariate_keys=continuous_covariate_keys,
        size_factor_key=size_factor_key,
    )  # X contains raw counts

    scvi_path = model_path / model_name
    if scvi_path.exists() and not retrain:
        vae = SCVI.load(scvi_path.as_posix(), adata.copy())

    else:
        training_kwargs = {}
        vae = SCVI(
            adata,
            n_layers=2,
            encode_covariates=False,  # True
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none",
        )  # .cuda()
        vae.train(
            max_epochs=scvi_epochs,
            train_size=0.85,
            batch_size=batch_size,
            early_stopping=True,
            **training_kwargs,
        )

    adata.obsm[SCVI_LATENT_KEY] = vae.get_latent_representation()

    if retrain or not scvi_path.exists():
        # save the reference model
        vae.save(scvi_path, overwrite=True)

    return vae, adata


def get_trained_scanvi(
    adata: AnnData,
    vae: SCVI | None = None,
    labels_key: str = "cell_type",
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "scanvi",
    **training_kwargs,
) -> (SCANVI, AnnData):
    """
    Get scANVI model and add latent representation to adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    scanvi_model : SCANVI
        An scANVI model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCANVI`.

    Returns
    -------
    SCANVI
        scANVI model.
    AnnData
        Annotated dat


    a matrix with latent variables.

    """
    scvi_epochs = 200
    batch_size = 512

    # n_labels = len(adata.obs[labels_key].cat.categories)
    batch_key = training_kwargs.pop("batch_key", None)

    if vae is None:
        # get the scVI model.  Note the desired batch_key needs to be passed
        # BUG: this will overwrite the SCVI model... need to fix
        scvi_model_name = "SCVI" + model_name
        vae, adata = get_trained_scvi(
            adata,
            labels_key=labels_key,
            batch_key=batch_key,
            model_path=model_path,
            retrain=False,
            model_name=scvi_model_name,
            **training_kwargs,
        )

    scanvi_path = model_path / model_name
    if scanvi_path.exists() and not retrain:
        scanvi_model = SCANVI.load(scanvi_path.as_posix(), adata)
    else:
        scanvi_model = SCANVI.from_scvi_model(vae, "Unknown", labels_key=labels_key)
        scanvi_model.train(
            max_epochs=scvi_epochs,
            train_size=0.85,
            batch_size=batch_size,
            early_stopping=True,
            **training_kwargs,
        )
    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)

    if retrain or not scanvi_path.exists():
        # save the reference model
        scanvi_model.save(scanvi_path, overwrite=True)

    return scanvi_model, adata


def get_query_scvi(
    adata: AnnData,
    vae: SCVI,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "query_scvi",
    **training_kwargs,
) -> (SCVI, AnnData):
    """
    Get scVI model via `scarches` surgery and add latent representation to adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    vae : SCVI
        An scVI model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    batch_key : str
        Key for batch labels. Default is `None`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.

    Returns
    -------
    SCVI
        scVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    surgery_epochs = 150

    qscvi_path = model_path / model_name

    SCVI.prepare_query_anndata(adata, vae)
    if qscvi_path.exists() and not retrain:
        scvi_query = SCVI.load(qscvi_path.as_posix(), adata)
    else:
        scvi_query = SCVI.load_query_data(adata, vae)
        scvi_query.train(
            max_epochs=surgery_epochs,
            train_size=0.85,
            early_stopping=True,
            plan_kwargs=dict(weight_decay=0.0),
            # datasplitter_kwargs=dict(num_workers=15),
            # **training_kwargs,
        )

    adata.obsm[SCVI_LATENT_KEY] = scvi_query.get_latent_representation(adata)
    if retrain or not qscvi_path.exists():
        # save the reference model
        scvi_query.save(qscvi_path, overwrite=True)

    return scvi_query, adata


def get_query_scanvi(
    adata: AnnData,
    scanvi_model: SCANVI,
    labels_key: str = "cell_type",
    model_path: Path = Path.cwd(),
    retrain: bool = False,
    model_name: str = "query_scanvi",
    **training_kwargs,
) -> (SCANVI, AnnData):
    """
    Get scANVI model via `scarches` surgery and add latent representation to adata.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    scanvi_model : SCANVI
        A scANVI model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    model_path : Path
        Path to save model. Default is `Path.cwd()`.
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `SCANVI_query`.

    Returns
    -------
    SCANVI
        scANVI model.
    AnnData
        Annotated data matrix with latent variables.

    """
    surgery_epochs = 150

    qscanvi_path = model_path / model_name
    SCANVI.prepare_query_anndata(adata, scanvi_model)

    if qscanvi_path.exists() and not retrain:
        scanvi_query = SCANVI.load(qscanvi_path.as_posix(), adata)
    else:
        scanvi_query = SCANVI.load_query_data(adata, scanvi_model)
        scanvi_query.train(
            max_epochs=surgery_epochs,
            train_size=0.85,
            early_stopping=True,
            plan_kwargs=dict(weight_decay=0.0),
            # datasplitter_kwargs=dict(num_workers=15),
        )

    adata.obsm[SCVI_LATENT_KEY] = scanvi_query.get_latent_representation(adata)

    if retrain or not qscanvi_path.exists():
        # save the reference model
        scanvi_query.save(qscanvi_path, overwrite=True)

    return scanvi_query, adata


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

    return vae, adata


def get_lbl8r(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "lbl8r",
    plot_training: bool = False,
    save: bool | Path | str = False,
    show: bool = True,
    fig_dir: Path | str | None = None,
    **training_kwargs,
):
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
    retrain : bool
        Whether to retrain the model. Default is `False`.
    model_name : str
        Name of the model. Default is `lbl8r`.
    plot_training : bool
        Whether to plot training. Default is `False`.
    save : bool | Path | str
        Whether to save the model. Default is `False`.
    show : bool
        Whether to show the plot. Default is `True`.
    fig_dir : Path|str|None
        Path to save the figure. Default is `None`.
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

    # not sure I need this step
    LBL8R.setup_anndata(adata, labels_key=labels_key)

    # 1. load/train model
    if lbl8r_path.exists() and not retrain:
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

    if retrain or not lbl8r_path.exists():
        # save the reference model
        lat_lbl8r.save(lbl8r_path, overwrite=True)

    if plot_training:
        plot_lbl8r_training(lat_lbl8r.history, save=save, show=show, fig_dir=fig_dir)

    return lat_lbl8r, adata


# TODO: modularize things better so the pca/scvi versions call same code
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
    pca_lbl8r, adata = get_lbl8r(
        adata,
        labels_key=labels_key,
        model_path=model_path,
        retrain=retrain,
        model_name=model_name,
        **training_kwargs,
    )

    return pca_lbl8r, adata


def get_xgb(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "xgb",
    **training_kwargs,
) -> (Booster, AnnData, LabelEncoder):
    """
    Load or train an XGBoost model and return the model, label encoder, and adata with predictions

    """
    PRED_KEY = "pred"
    # format model_path / model_name for xgboost
    bst_path = (
        model_path / model_name
        if model_name.endswith(".json")
        else model_path / f"{model_name}.json"
    )

    labels_key = labels_key
    n_labels = len(adata.obs[labels_key].cat.categories)

    X_train, y_train, label_encoder = get_xgb_data(adata, label_key=labels_key)
    use_gpu = training_kwargs.pop("use_gpu", True)
    if bst_path.exists() and not retrain:
        # load trained model''
        print(f"loading {bst_path}")
        bst = load_xgboost(bst_path, use_gpu=use_gpu)
    else:
        bst = None

    if bst is None:
        print(f"training {model_name}")
        # train
        bst = train_xgboost(X_train, y_train)

    if retrain or not bst_path.exists():
        # save the reference model
        bst.save_model(bst_path)
        # HACK: reload to so that the training GPU memory is cleared
        bst = load_xgboost(bst_path, use_gpu=use_gpu)
        print("reloaded bst (memory cleared?)")

    return bst, adata, label_encoder


## wrappers to load data
import anndata as ad
from scvi.model import SCVI, SCANVI
from pathlib import Path

from xgboost import Booster
from sklearn.preprocessing import LabelEncoder

from .model._lbl8r import LBL8R
from .utils import (
    make_latent_adata,
    # add_predictions_to_adata,
    # merge_into_obs,
    # query_scanvi,
    # plot_scvi_training,
    # plot_scanvi_training,
    # plot_lbl8r_training,
    make_pc_loading_adata,
)

from ._constants import *
from .model._xgb import train_xgboost, test_xgboost, load_xgboost, get_xgb_data
from ._artifacts import Adata

"""

Loading should

0. probe the path to see if an "out" version exists
    - if so, load it and return it
    - if not, load the raw data and prep it

1. defone the data paths
    - input.
    - outut. 
    - exptra.  e.g. expr version

2. prep the data.
2. load the data



"""


# wrap adata into a dataclass
def load_adata(
    adata_path: str | Path,
) -> Adata:
    """
    Load data for training wrapped in Adata for easy artifact generation
    """
    adata_path = Path(adata_path)
    return Adata(adata_path)


def load_training_data(adata_path: str | Path) -> Adata:
    """
    Load training data.
    """
    adata_path = Path(adata_path)
    # warn if "train" is not in data_path.name
    if "train" not in adata_path.name:
        print(
            f"WARNING:  'train' not in data_path.name: {adata_path.name}.  "
            "This may cause problems with the output file names."
        )

    return Adata(adata_path)


def load_query_data(adata_path: str | Path) -> Adata:
    """
    Load query data.
    """

    adata_path = Path(adata_path)
    # warn if "train" is not in data_path.name
    if "test" not in adata_path.name:
        print(f"WARNING:  'train' not in data_path.name: {adata_path.name}.  ")

    return Adata(adata_path)
