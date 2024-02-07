from xgboost import Booster
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from anndata import AnnData
from numpy import unique, asarray, argmax
import pickle
from pandas import DataFrame
import numpy as np
import pandas as pd

from .utils._timing import Timing
from .utils._device import get_usable_device
from .utils._le import load_label_encoder


class XGB:
    """
    XGBoost model class for storing models + metadata.
    """

    def __init__(
        self,
        adata: AnnData | None = None,
        n_labels: int | None = None,
        input_type: str | None = None,
        **kwargs,
    ):
        self.n_labels = n_labels

        self._module: xgb.Booster | None = None
        self._label_encoder = None
        self._path: Path | str | None = None
        self._name: str | None = None
        self.adata: AnnData | None = adata
        self.X = None
        self.y = None

        self.n_labels = n_labels
        self.module = xgb.Booster()

        self._model_summary_string = (
            "XGB classifier model"
            if input_type is None
            else f"{input_type} XGB classifier model"
        )
        # self.init_params_ = self._get_init_params(locals())

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        if not "xgb" in name:
            print(f"Error:  {name} is not an XGBoost model.")
        else:
            self._name = name

    @property
    def label_encoder(self):
        if self._label_encoder is None and self.path is not None:
            self._label_encoder = load_label_encoder(self.path.parent)

        return self._label_encoder

    @label_encoder.setter
    def label_encoder(self, label_encoder: LabelEncoder):
        self._label_encoder = label_encoder

    # make module lazy ? not sure about this logic
    @property
    def module(self):
        if self._module is None:
            if self.path is not None:
                self._module = load_xgboost(self.path)
            else:
                print("Error:  No path set.")

        return self._module

    @module.setter
    def module(self, module: Booster):
        self._module = module

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Path):
        # check if there's a model in the path
        # also set _trained to True if there is
        # or set _trained to False if there isn't
        if path is not None:
            if path.exists():
                if path.is_file():
                    self._path = path
                else:
                    self._path = path / "xgb.json"
                self.name = path.parent.name
            else:
                path.mkdir(exist_ok=True, parents=True)
                self._path = path / "xgb.json"
                self.name = path.parent.name

    # wrapper for instantiating model from disk.
    def load(self, path: Path | str):
        """
        Load an XGBoost classifier model from a file.

        Parameters
        ----------
        path : Path | str
            The file path to the saved XGBoost model. i.e. 'path/to/xgb.json'

        Returns
        -------
        model: Booster
            The loaded XGBoost classifier model, or None if loading fails.
        """

        # Check if the file exists
        path = Path(path) if isinstance(path, str) else path

        if not path.exists():
            print(f"Error: The file '{path}' does not exist.")
            return None

        self.path = path

        print(f"loading xgb model from: {self.path}")
        # Set the global configuration for XGBoost
        try:
            self.label_encoder = load_label_encoder(self.path.parent)

            model = xgb.Booster()
            model.load_model(self.path)

            # self.label_encoder = LabelEncoder().load(path.parent / "label_encoder.pkl")

            self._module = model

            return self

        except Exception as e:
            # Handle exceptions that occur during model loading
            print(f"Failed to load the model: {e}")
            return None

    def prep_adata(self, adata: AnnData | None = None, label_key: str = "cell_type"):
        """
        consider adding a "pre-processing" step to normalize the data
        """
        if adata is None:
            adata = self.adata

        X = adata.X
        y = adata.obs[label_key]
        if self.label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder = label_encoder.fit(y)
            self.label_encoder = label_encoder

        y = self.label_encoder.transform(y)

        return X, y

    def train(
        self,
        labels_key: str = "cell_type",
        **training_kwargs,
    ) -> (Booster, AnnData, LabelEncoder):
        """
        make a booster model and train it


        """

        PRED_KEY = "pred"
        # format path / model_name for xgboost

        labels_key = labels_key
        n_labels = len(self.adata.obs[labels_key].cat.categories)

        X_train, y_train = self.prep_adata(label_key=labels_key)

        print(f"training {self.name}")
        # train
        self.module = train_xgboost2(X_train, y_train, **training_kwargs)

        return self

    def predict(
        self, adata: AnnData, label_key: str = "cell_type", report: bool = False
    ):
        """ """
        # TODO: add a total size check and iteratively predict if its too large

        X_test = adata.X
        le = self.label_encoder
        y_test = le.transform(adata.obs[label_key])
        index = adata.obs.index

        num_samples = X_test.shape[0]
        chunk_size = 10000  # Define a chunk size that fits in your memory
        its = 0
        preds = []
        # HACK: this is a hack to deal with large datasets. should do at once if possible
        for i in range(0, num_samples, chunk_size):
            # Read a chunk of data
            X_chunk = X_test[i : i + chunk_size, :]
            y_chunk = y_test[i : i + chunk_size]

            dtest = xgb.DMatrix(X_chunk, label=y_chunk)
            preds.append(self.module.predict(dtest))

            its += 1
        preds = np.vstack(preds)
        print(f" ending chunk size = {X_chunk.shape} in {its=} iterations")

        # dtest = xgb.DMatrix(X_test, label=y_test)
        # preds = self.module.predict(dtest)

        classes = self.label_encoder.classes_
        # Predict the probabilities for each class on the test set

        # Convert the predictions into class labels
        best_preds = asarray([argmax(line) for line in preds])

        # TODO: deal with this report properly...
        # Evaluate the model on the test set
        report = classification_report(
            y_test, best_preds, target_names=classes, output_dict=True
        )

        # Convert predictions to DataFrame
        preds_df = DataFrame(preds, columns=classes, index=index)

        # Convert the predictions into class labels
        # there's a pandas argmax way thats cleaner
        best_preds = asarray([argmax(line) for line in preds])
        # Add actual and predicted labels to DataFrame
        preds_df["label"] = self.label_encoder.inverse_transform(best_preds)

        # # Evaluate the model on the test set
        # print(classification_report(y_test, best_preds, target_names=classes))
        if report:
            return preds_df, report
        else:
            return preds_df  # , report

    def save(self, save_path: Path | str | None = None):
        """ """
        if save_path is None:
            save_path = self.path
        else:
            self.path = save_path
        print(f"Saving the model to '{self.path}'.")

        # save the reference model
        self.module.save_model(self.path)
        # self.label_encoder.save(self.path.parent / "label_encoder.pkl")
        # Save the LabelEncoder to a file for easy query access
        with open(self.path.parent / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        print(f"Saved the model to '{self.path}'.")


@Timing(prefix="model_name")
def get_xgb2(
    adata: AnnData,
    labels_key: str = "cell_type",
    model_path: Path = ".",
    retrain: bool = False,
    model_name: str = "xgb",
    **training_kwargs,
) -> tuple[XGB, AnnData]:
    n_labels = len(adata.obs[labels_key].cat.categories)

    retrain = True
    # 1. load/train model
    if model_path.exists() and not retrain:
        bst = XGB()
        bst.load(model_path / model_name)
    else:
        bst = XGB(adata, n_labels=n_labels)
        bst.train(**training_kwargs)

    if retrain or not model_path.exists():
        # save the reference model
        bst.save(model_path / model_name)

    return bst, adata


@Timing(prefix="model_name")
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
    bst_path = model_path / model_name

    n_labels = len(adata.obs[labels_key].cat.categories)

    X_train, y_train, label_encoder = get_xgb_data(adata, label_key=labels_key)

    if bst_path.exists() and not retrain:
        # load trained model''
        print(f"loading {bst_path}")
        bst, le = load_xgboost(bst_path)
    else:
        bst = None

    if bst is None:
        print(f"training {model_name}")
        # train
        bst = train_xgboost(X_train, y_train, **training_kwargs)

    if retrain or not bst_path.exists():
        save_xgboost(bst, bst_path, label_encoder)
        # HACK: reload to so that the training GPU memory is cleared
        bst, _ = load_xgboost(bst_path)
        print("reloaded bst (memory cleared?)")

    return bst, adata, label_encoder


def train_xgboost(X, y, num_round=50, **training_kwargs) -> xgb.Booster:
    """
    wrapper to split validation set and train xgboost and train model
    """

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    n_cats = len(unique(y))

    device = training_kwargs.pop("device", None)

    device = get_usable_device(device)

    max_depth = training_kwargs.pop("max_depth", 7)
    objective = training_kwargs.pop("objective", "multi:softprob")
    eta = training_kwargs.pop("eta", 0.3)

    params = dict(
        max_depth=max_depth,
        objective=objective,
        num_class=n_cats,
        eta=eta,
        device=device,
    )

    bst = xgb.train(
        params,
        dtrain,
        num_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=10,
        verbose_eval=10,
    )
    return bst


def train_xgboost2(
    X, y, num_round: int = 50, chunk_size: int = 10000, **training_kwargs
) -> xgb.Booster:
    """
    wrapper to split validation set and train xgboost and train model
    """

    num_samples = X.shape[0]

    n_cats = len(unique(y))
    device = training_kwargs.pop("device", None)

    device = get_usable_device(device)

    max_depth = training_kwargs.pop("max_depth", 7)
    objective = training_kwargs.pop("objective", "multi:softprob")
    eta = training_kwargs.pop("eta", 0.3)

    params = dict(
        max_depth=max_depth,
        objective=objective,
        num_class=n_cats,
        eta=eta,
        device=device,
    )

    # overlap the chunks?
    for i in range(0, num_samples, chunk_size // 2):
        # Read a chunk of data
        # X_chunk = f['features'][i:i + chunk_size]
        # y_chunk = f['labels'][i:i + chunk_size]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X[i : i + chunk_size, :], y[i : i + chunk_size], test_size=0.15
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)

        if i == 0:
            bst = xgb.train(
                params,
                dtrain,
                num_round,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=10,
                verbose_eval=10,
            )
        # Otherwise, continue training with the existing model
        else:
            bst = xgb.train(
                params,
                dtrain,
                num_round,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=10,
                verbose_eval=10,
                xgb_model=bst,
            )

    return bst


def save_xgboost(bst: Booster, bst_path: Path, label_encoder: LabelEncoder):
    """
    Save an XGBoost classifier model to a file.

    Parameters
    ----------
    bst : Booster
        The XGBoost classifier model.
    bst_path : Path
        The file path to save the XGBoost model. i.e. 'path/to/xgb.json'
    """
    if not bst_path.is_file():
        bst_path = bst_path / "xgb.json"

    bst_path.parent.mkdir(exist_ok=True, parents=True)
    # save the reference model
    bst.save_model(bst_path)
    label_encoder.save(bst_path.parent / "label_encoder.pkl")
    print(f"Saved the model to '{bst_path}'.")


def get_xgb_data(adata, label_key="cell_type"):
    """
    consider adding a "pre-processing" step to normalize the data
    """
    X = adata.X
    y = adata.obs[label_key]
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    y = label_encoder.transform(y)
    return X, y, label_encoder


def load_xgboost(bst_path: Path | str) -> xgb.Booster | None:
    """
    Load an XGBoost classifier model from a file.

    Parameters
    ----------
    bst_path : Path | str
        The file path to the saved XGBoost model. i.e. 'path/to/xgb.json'

    Returns
    -------
    model: Booster
        The loaded XGBoost classifier model, or None if loading fails.
    """

    # Check if the file exists
    bst_path = Path(bst_path) if isinstance(bst_path, str) else bst_path

    if not bst_path.is_file():
        bst_path = bst_path / "xgb.json"

    if not bst_path.exists():
        print(f"Error: The file '{bst_path}' does not exist.")
        return None

    # Set the global configuration for XGBoost
    try:
        # # Load the model
        # if use_gpu:
        #     model = xgb.Booster(predictor="gpu_predictor")
        # else:
        #     model = xgb.XGBClassifier(predictor="cpu_predictor")
        model = xgb.Booster()
        model.load_model(bst_path)

        label_encoder = LabelEncoder().load(bst_path.parent / "label_encoder.pkl")
        return model, label_encoder

    except Exception as e:
        # Handle exceptions that occur during model loading
        print(f"Failed to load the model: {e}")
        return None, None


# TODO:  query_xgb should be query_xgboost
def query_xgb(
    adata: AnnData,
    bst: XGB,
) -> pd.DataFrame:
    # ) -> AnnData:
    """
    Test the XGBoost classifier on the test set

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    labelator : Booster
        An XGBoost classification model.
    label_encoder : LabelEncoder
        The label encoder.
    labels_key : str
        Key for cell type labels. Default is `cell_type`.

    Returns
    -------
    AnnData
        Annotated data matrix with latent variables as X

    """
    # predictions = bst.predict(adata, label_key="cell_type")
    predictions, report = bst.predict(adata, label_key="cell_type", report=True)

    return predictions, report


def query_xgboost(
    bst: xgb.Booster,
    adata: AnnData,
    label_encoder: LabelEncoder,
    label_key: str = "cell_type",
) -> (DataFrame, dict):
    """
    Test the XGBoost classifier on the test set.

    Parameters:
    bst: xgboost.Booster
        The XGBoost classifier model.
    adata: AnnData
        The test dataset.
    label_encoder: LabelEncoder
        The label encoder.
    label_key: str
        The key in adata.obs that contains the labels.

    Returns:
    preds_df: DataFrame
        The predictions for each class.
    report: dict
        The classification report.
    """
    X_test = adata.X
    y_test = label_encoder.transform(adata.obs[label_key])
    index = adata.obs.index
    dtest = xgb.DMatrix(X_test, label=y_test)

    classes = label_encoder.classes_
    # Predict the probabilities for each class on the test set
    preds = bst.predict(dtest)

    # Convert the predictions into class labels
    best_preds = asarray([argmax(line) for line in preds])

    # Evaluate the model on the test set
    report = classification_report(
        y_test, best_preds, target_names=classes, output_dict=True
    )

    # Convert predictions to DataFrame
    preds_df = DataFrame(preds, columns=classes, index=index)

    # Convert the predictions into class labels
    # there's a pandas argmax way thats cleaner
    best_preds = asarray([argmax(line) for line in preds])
    # Add actual and predicted labels to DataFrame
    preds_df["label"] = label_encoder.inverse_transform(best_preds)

    # # Evaluate the model on the test set
    # print(classification_report(y_test, best_preds, target_names=classes))

    return preds_df, report
