## wrappers for getting trained models
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
import scanpy as sc

from .model._lbl8r import (
    LBL8R,
    get_lbl8r,
    query_lbl8r,
    get_scvi_lbl8r,
    get_pca_lbl8r,
    prep_pcs_adata,
)

from .model._scvi import (
    get_trained_scvi,
    get_trained_scanvi,
    get_query_scvi,
    get_query_scanvi,
    query_scanvi,
    prep_latent_z_adata,
    make_scvi_normalized_adata,
    make_latent_adata,
)

from .model._xgb import get_xgb, query_xgb
from .model.utils._data import Adata, transfer_pcs
from .model.utils._Model import Model
from .model.utils._plot import make_plots
from ._constants import *

MODEL = SCVI | SCANVI | LBL8R | Booster

PRED_KEY = "label"
INSERT_KEY = "pred"


# TODO:  make the code this so we can save tables instead of always writing anndata
def create_artifacts(data: list, model: list, visualization_path, artifacts_path):
    """
    Create the artifacts
    """

    # create the artifacts
    if visualization_path:
        # create the visualizations
        pass

    if artifacts_path:
        # create the artifacts
        pass


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
3. load the data
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


def prep_model(
    data: Adata,
    model_name: str,
    model_path: Path | str,
    model: SCVI | SCANVI | LBL8R | None = None,
    labels_key: str = "cell_type",
    # batch_key: str | None = None, # TODO:  model_kwargs for controlling batch_key, etc..
    retrain: bool = False,
    **training_kwargs,
) -> Model:
    """
    Load a model from path or train and prep data

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

    # model_path = model_path / model_name
    print(f"prep_model0: getting {(model_path/model_name)} model")

    if not model_path.exists():
        print(f"no model_path, will create and train model")
        # raise ValueError(f"{model_path} does not exist")

    batch_key = training_kwargs.pop("batch_key", None)

    # 0. tag adata for artifact export
    data.set_output(model_name)

    ad = data.adata
    ret_models = []
    # SCANVI E2E MODELS
    if model_name in (SCANVI_MODEL_NAME, SCANVI_BATCH_EQUALIZED_MODEL_NAME):
        batch_key = (
            "sample" if model_name == SCANVI_BATCH_EQUALIZED_MODEL_NAME else None
        )

        # create a ground truth for SCANVI so we can clobber the labels_key for queries
        ad.obs["ground_truth"] = ad.obs[labels_key].to_list()

        # load teh scvi model, scanvi_model, (qnd query models?)
        print(f"scanvi getting 0{(model_path/model_name/SCVI_SUB_MODEL_NAME)}")
        vae, ad = get_trained_scvi(
            ad,
            labels_key=labels_key,
            batch_key=batch_key,
            model_path=(model_path / model_name),
            retrain=retrain,
            model_name=SCVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        print(f"scanvi getting 1 {(model_path/model_name/SCANVI_SUB_MODEL_NAME)}")
        model, ad = get_trained_scanvi(
            ad,
            vae,
            labels_key=labels_key,
            model_path=(model_path / model_name),
            retrain=retrain,
            model_name=SCANVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        # update data with ad
        data.update(ad)

        model = Model(model, model_path, model_name, vae, labels_key)
        return model, data

    elif model_name in (
        SCVI_LATENT_MODEL_NAME,
        XGB_SCVI_LATENT_MODEL_NAME,
        LBL8R_SCVI_EXPRESION_MODEL_NAME,
        SCVI_EXPR_PC_MODEL_NAME,
        XGB_SCVI_EXPRESION_MODEL_NAME,
        XGB_SCVI_EXPR_PC_MODEL_NAME,
    ):
        # need vae
        print(
            f"getting scvi: 1 {(model_path/SCVI_LATENT_MODEL_NAME/SCVI_SUB_MODEL_NAME)}"
        )
        vae, ad = get_trained_scvi(
            ad,
            labels_key=labels_key,
            batch_key=None,
            model_path=(model_path / SCVI_LATENT_MODEL_NAME),
            retrain=retrain,
            model_name=SCVI_SUB_MODEL_NAME,
            **training_kwargs,
        )

        # SCVI LBL8R Model
        if model_name in (SCVI_LATENT_MODEL_NAME, XGB_SCVI_LATENT_MODEL_NAME):
            # 1. make the make_latent_adata
            ad = prep_latent_z_adata(ad, vae, labels_key=labels_key)
            # try to load a pre-trained model. ... check if it exists?
            if model_name == SCVI_LATENT_MODEL_NAME:
                print(
                    f"getting scvi lbl8r 2: {(model_path/model_name/LBL8R_SCVI_SUB_MODEL_NAME)}"
                )
                model, ad = get_lbl8r(
                    ad,
                    labels_key=labels_key,
                    model_path=(model_path / model_name),
                    retrain=retrain,
                    model_name=LBL8R_SCVI_SUB_MODEL_NAME,
                    **training_kwargs,
                )
                # 2. update the data with the latent representation
                data.update(ad)

                model = Model(
                    model,
                    model_path,
                    model_name,
                    vae=vae,
                    labels_key=labels_key,
                )
                return model, data

            elif model_name == XGB_SCVI_LATENT_MODEL_NAME:
                model, ad, label_encoder = get_xgb(
                    ad,
                    labels_key=labels_key,
                    model_path=model_path,
                    retrain=retrain,
                    model_name=model_name,
                    **training_kwargs,
                )
                # 2. update the data with the latent representation
                data.update(ad)

                model = Model(
                    model,
                    model_path,
                    model_name,
                    vae=vae,
                    labels_key=labels_key,
                    label_encoder=label_encoder,
                )
                return model, data

        elif model_name in (
            LBL8R_SCVI_EXPRESION_MODEL_NAME,
            XGB_SCVI_EXPRESION_MODEL_NAME,
            SCVI_EXPR_PC_MODEL_NAME,
            XGB_SCVI_EXPR_PC_MODEL_NAME,
        ):
            # TODO:  load this data rather than compute it... saves 45s
            # 1. make scvi_expression data
            ad = make_scvi_normalized_adata(vae, ad)
            # 2. update the data with pcs
            sc.pp.pca(ad)
            # 3. update the data with the latent representation
            # ad.obsm["X_scVI"] = vae.get_latent_representation(ad)

            if model_name == LBL8R_SCVI_EXPRESION_MODEL_NAME:
                model, ad = get_lbl8r(
                    ad,
                    labels_key=labels_key,
                    model_path=model_path,
                    retrain=retrain,
                    model_name=model_name,
                    **training_kwargs,
                )
                # 2. update the data with the latent representation
                data.update(ad)
                model = Model(
                    model,
                    model_path,
                    model_name,
                    vae=vae,
                    labels_key=labels_key,
                )
                return model, data

            elif model_name == XGB_SCVI_EXPRESION_MODEL_NAME:
                print(f"getting xgb scvi expr {(model_path/model_name)}")
                model, ad, label_encoder = get_xgb(
                    ad,
                    labels_key=labels_key,
                    model_path=model_path,
                    retrain=retrain,
                    model_name=model_name,
                    **training_kwargs,
                )
                # 2. update the data with the latent representation
                data.update(ad)
                model = Model(
                    model,
                    model_path,
                    model_name,
                    vae=vae,
                    labels_key=labels_key,
                    label_encoder=label_encoder,
                )
                return model, data

            elif model_name in (SCVI_EXPR_PC_MODEL_NAME, XGB_SCVI_EXPR_PC_MODEL_NAME):
                # 1. make the pcs representation
                ad = prep_pcs_adata(ad, pca_key=PCA_KEY)
                if model_name == SCVI_EXPR_PC_MODEL_NAME:
                    print(f"getting scvi expr pcs {(model_path/model_name)}")
                    model, ad = get_lbl8r(
                        ad,
                        labels_key=labels_key,
                        model_path=model_path,
                        retrain=retrain,
                        model_name=model_name,
                        **training_kwargs,
                    )
                    # 2. update the data with the latent representation
                    data.update(ad)
                    model = Model(
                        model, model_path, model_name, vae=vae, labels_key=labels_key
                    )
                    return model, data

                elif model_name == XGB_SCVI_EXPR_PC_MODEL_NAME:
                    print(f"getting xgb scvi expr pcs {(model_path/model_name)}")
                    model, ad, label_encoder = get_xgb(
                        ad,
                        labels_key=labels_key,
                        model_path=model_path,
                        retrain=retrain,
                        model_name=model_name,
                        **training_kwargs,
                    )
                    # 2. update the data with the latent representation
                    data.update(ad)
                    model = Model(
                        model,
                        model_path,
                        model_name,
                        vae=vae,
                        labels_key=labels_key,
                        label_encoder=label_encoder,
                    )
                    return model, data

            else:
                raise ValueError(f"IMPOSSIBLE unknown model_name {model_name}")

    elif model_name in (RAW_PC_MODEL_NAME, XGB_RAW_PC_MODEL_NAME):
        print(f"getting raw pcs {(model_path/model_name)}")

        # 1. get the pcs representation
        ad = prep_pcs_adata(ad, pca_key=PCA_KEY)
        # 2. update the data with the latent representation
        data.update(ad)
        if model_name == RAW_PC_MODEL_NAME:
            model, ad = get_lbl8r(
                ad,
                labels_key=labels_key,
                model_path=model_path,
                retrain=retrain,
                model_name=model_name,
                **training_kwargs,
            )
            # 2. update the data with the latent representation
            data.update(ad)
            model = Model(model, model_path, model_name, labels_key=labels_key)
            return model, data

        elif model_name == XGB_RAW_PC_MODEL_NAME:
            print(f"getting xgb scvi expr pcs {(model_path/model_name)}")
            model, ad, label_encoder = get_xgb(
                ad,
                labels_key=labels_key,
                model_path=model_path,
                retrain=retrain,
                model_name=model_name,
                **training_kwargs,
            )
            # 2. update the data with the latent representation
            data.update(ad)
            model = Model(
                model,
                model_path,
                model_name,
                labels_key=labels_key,
                label_encoder=label_encoder,
            )
            return model, data

    # other e2e models
    elif model_name == LBL8R_RAW_COUNT_MODEL_NAME:
        # assume path for expression data variant already gets the right data
        print(f"getting raw lbl8r {(model_path/model_name)}")
        model, ad = get_lbl8r(
            ad,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )
        # 2. update the data with the latent representation
        data.update(ad)
        model = Model(model, model_path, model_name, labels_key=labels_key)
        return model, data

    elif model_name == XGB_RAW_COUNT_MODEL_NAME:
        # assume path for expression data variant already gets the right data
        print(f"getting raw xgb {(model_path/model_name)}")
        model, ad, label_encoder = get_xgb(
            ad,
            labels_key=labels_key,
            model_path=model_path,
            retrain=retrain,
            model_name=model_name,
            **training_kwargs,
        )
        # 2. update the data with the latent representation
        data.update(ad)
        model = Model(
            model,
            model_path,
            model_name,
            labels_key=labels_key,
            label_encoder=label_encoder,
        )
        return model, data

    else:
        raise ValueError(f"unknown model_name {model_name}")


def prep_latent_data(data: Adata, vae: SCVI, labels_key: str = "cell_type") -> Adata:
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    model :
        An classification model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    # 1. get the latent representation
    ad = prep_latent_z_adata(data.adata, vae=vae, labels_key=labels_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def prep_expr_data(data: Adata, vae: SCVI, ref_data: Adata | None = None) -> Adata:
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    vae :
        An scvi model used for normalization.
    ref_data : Adata
        "training" data used for transferring pcs

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    # 1. get the normalized adata
    ad = make_scvi_normalized_adata(vae, data.adata)

    # 2. update the data with pcs
    if ref_data is None:
        ValueError("ref_data is None, can't make scvi normalized pcs")
    else:
        ref = ref_data.adata
        ad = transfer_pcs(ref, ad)

    # 3. update the data with the latent representation & pcs
    data.update(ad)
    return data


def prep_pc_data(data: Adata, pca_key=PCA_KEY, ref_data: Adata | None = None) -> Adata:
    """
    Prep adata for pcs LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    pca_key : str
        Key for pca. Default is `X_pca`.
    ref_data : Adata
        "training" data used for transferring pcs

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    ad = data.adata
    # 1. make sure we have pcs
    if ad.varm.get("PCs") is None:
        if ref_data is None:
            ValueError("ref_data is None, can't transfer PCs")
        else:
            # check ref_data.adat.varm and ref_data.adata.uns for "PCs"
            ref = ref_data.adata
            ad = transfer_pcs(ref, ad)

    # 1. get the pcs representation
    ad = prep_pcs_adata(ad, pca_key=pca_key)
    # 2. update the data with the latent representation
    data.update(ad)
    return data


def query_model(
    data: Adata,
    model: Model,
) -> Adata:
    """
    Attach a classifier and prep adata for scVI LBL8R model

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    model : LBL8R | SCANVI | Booster
        An classification model.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """

    if isinstance(model.model, SCANVI):
        # "transfer learning" query models which need to be trained
        ad = query_scanvi(data.adata, model.model)

        # fix the labels_key with "ground_truth"
        ad.obs[model.labels_key] = ad.obs["ground_truth"].to_list()

    # no prep needed to query these models.
    elif isinstance(model.model, LBL8R):
        ad = query_lbl8r(data.adata, model.model)
    elif isinstance(model.model, Booster):
        ad, report = query_xgb(
            data.adata, model.model, label_encoder=model.label_encoder
        )
        # TODO: do something with the report...
    else:
        raise ValueError(f"model {model} is not a valid model")

    # update data with ad
    data.update(ad)

    return data


def prep_query_scanvi(
    data: Adata,
    model: Model,
    labels_key: str = "cell_type",
    retrain: bool = False,
) -> (Adata, Model):
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    model :
        An classification model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    retrian : bool
        Retrain the model. Default is `False`.

    Returns
    -------
    Adata
        updated data
    Model
        Model with .q_vae, and .scanvi models added and querey_scanvi model as .model
    """

    ad = data.adata

    # create a ground truth for SCANVI so we can clobber the labels_key for queries
    ad.obs["ground_truth"] = ad.obs[labels_key].to_list()
    ad.obs[labels_key] = "Unknown"

    # 1. get query_scvi (depricate?  needed for what?  latent conditioning?)
    q_scvi, ad = get_query_scvi(
        ad,
        model.vae,
        labels_key=model.labels_key,
        model_path=(model.path / model.name),
        retrain=retrain,
        model_name=QUERY_SCVI_SUB_MODEL_NAME,
    )
    # 2. get query_scanvi
    q_scanvi, ad = get_query_scanvi(
        ad,
        model.model,
        labels_key=model.labels_key,
        model_path=(model.path / model.name),
        retrain=retrain,
        model_name=QUERY_SCANVI_SUB_MODEL_NAME,
    )
    # update data with ad
    data.update(ad)

    # 3. return model, pack all the models into the Model
    model.scanvi = model.model
    model.q_vae = q_scvi
    model.model = q_scanvi

    return data, model


def prep_query_model(
    query_data: Adata,
    model: Model,
    model_name: str,
    ref_data: Adata,
    labels_key: str = "cell_type",
    retrain: bool = False,
):
    """
    Prep adata for scVI LBL8R model

    Parameters
    ----------
    data : Adata
        dataclass holder for Annotated data matrix.
    model :
        An classification model.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.
    retrian : bool
        Retrain the model. Default is `False`.

    Returns
    -------
    Adata
        Annotated data matrix with latent variables as X

    """

    # 0. tag adata for artifact export
    query_data.set_output(model_name)

    # 1. prep query data (normalize / get latents / transfer PCs (if normalized) )
    if model_name in (
        LBL8R_SCVI_EXPRESION_MODEL_NAME,
        XGB_SCVI_EXPRESION_MODEL_NAME,
        SCVI_EXPR_PC_MODEL_NAME,
        XGB_SCVI_EXPR_PC_MODEL_NAME,
    ):
        # add latent representation to query_data?
        # ad.obsm["X_scVI"] = model.vae.get_latent_representation(query_data.data)
        # query_data.update(ad)

        # SCVI expression models
        query_data = prep_expr_data(query_data, model.vae, ref_data=ref_data)
        if model_name in (
            SCVI_EXPR_PC_MODEL_NAME,
            XGB_SCVI_EXPR_PC_MODEL_NAME,
        ):
            query_data = prep_pc_data(query_data, ref_data=ref_data)

    elif model_name in (
        SCVI_LATENT_MODEL_NAME,
        XGB_SCVI_LATENT_MODEL_NAME,
    ):
        # SCVI embedding models
        query_data = prep_latent_data(query_data, model.vae, labels_key=labels_key)

    elif model_name in (
        RAW_PC_MODEL_NAME,
        XGB_RAW_PC_MODEL_NAME,
    ):
        # PCS models
        query_data = prep_pc_data(query_data, ref_data=ref_data)

    elif model_name in (
        SCANVI_BATCH_EQUALIZED_MODEL_NAME,
        SCANVI_MODEL_NAME,
    ):
        # SCANVI models
        query_data, model = prep_query_scanvi(
            query_data,
            model,
            labels_key=labels_key,
            retrain=retrain,
        )

    return query_data, model


def archive_plots(
    data: Adata,
    model: Model,
    train_or_query: str,
    labels_key: str,
    path: Path | str | None = None,
):
    """
    Archive the plots
    """
    figs = make_plots(data, model, train_or_query, labels_key=labels_key, path=path)

    ## save plots..
    for fig in figs:
        fig.savefig()


def archive_data(
    data: Adata,
    path: Path,
):
    """
    Archive the data
    """

    data.export(path)
