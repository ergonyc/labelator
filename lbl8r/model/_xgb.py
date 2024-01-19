from xgboost import Booster
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from anndata import AnnData
from numpy import unique, asarray, argmax

from pandas import DataFrame

from .utils._data import merge_into_obs


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
    if model_name.endswith(".json"):
        bst_path = model_path / model_name
    else:
        bst_path = model_path / model_name / "xgb.json"

    labels_key = labels_key
    n_labels = len(adata.obs[labels_key].cat.categories)

    X_train, y_train, label_encoder = get_xgb_data(adata, label_key=labels_key)

    if bst_path.exists() and not retrain:
        # load trained model''
        print(f"loading {bst_path}")
        bst = load_xgboost(bst_path)
    else:
        bst = None

    if bst is None:
        print(f"training {model_name}")
        # train
        bst = train_xgboost(X_train, y_train, **training_kwargs)

    if retrain or not bst_path.exists():
        # save the reference model
        bst.save_model(bst_path)
        # HACK: reload to so that the training GPU memory is cleared
        bst = load_xgboost(bst_path)
        print("reloaded bst (memory cleared?)")

    return bst, adata, label_encoder


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


def train_xgboost(X, y, num_round=50, **training_kwargs) -> xgb.Booster:
    """
    wrapper to split validation set and train xgboost and train model
    """

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    n_cats = len(unique(y))

    use_gpu = training_kwargs.pop("use_gpu", False)
    max_depth = training_kwargs.pop("max_depth", 7)
    objective = training_kwargs.pop("objective", "multi:softprob")
    eta = training_kwargs.pop("eta", 0.3)
    device = "cuda" if use_gpu else None

    params = dict(
        max_depth=max_depth,
        objective=objective,
        num_class=n_cats,
        eta=eta,
        device=device,
    )

    # params = {
    #     "max_depth": 7,
    #     "objective": "multi:softprob",  # error evaluation for multiclass training
    #     "num_class": n_cats,
    #     "eta": 0.3,  # the training step for each iteration
    #     # 'n_gpus': 0
    # }
    # if use_gpu:
    #     params["device"] = "cuda"

    # # Set up parameters for xgboost
    # param = {
    # 'max_depth': 6,  # the maximum depth of each tree
    # 'eta': 0.3,  # the training step for each iteration
    # 'objective': 'multi:softprob',  # error evaluation for multiclass training
    # 'num_class': len(np.unique(y_train_full_encoded)) } # the number of classes

    # evallist = [(dval, 'eval'), (dtrain, 'train')]
    # bst = xgb.train(param, dtrain, num_round, evallist)
    bst = xgb.train(
        params,
        dtrain,
        num_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=10,
        verbose_eval=10,
    )
    return bst


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

        return model
    except Exception as e:
        # Handle exceptions that occur during model loading
        print(f"Failed to load the model: {e}")
        return None


# TODO:  query_xgb should be query_xgboost
def query_xgb(
    adata: AnnData,
    bst: Booster,
    label_encoder: LabelEncoder,
) -> (AnnData, dict):
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

    predictions, report = query_xgboost(bst, adata, label_encoder)
    # loadings_ad = add_predictions_to_adata(
    #     adata, predictions, insert_key=INSERT_KEY, pred_key=PRED_KEY
    # )
    adata = merge_into_obs(adata, predictions)

    return adata, report


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
