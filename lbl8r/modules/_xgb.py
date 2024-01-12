import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
from anndata import AnnData
from numpy import asarray, argmax, unique, array
from pandas import DataFrame
from typing import Any

def get_xgb_data(adata, label_key='cell_type'):
    """
    consider adding a "pre-processing" step to normalize the data
    """
    X = adata.X
    y = adata.obs[label_key]
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y)
    y = label_encoder.transform(y)
    return X, y, label_encoder

def train_xgboost(X,y,num_round = 50 , use_gpu=True) -> xgb.Booster: 
    """
    wrapper to split validation set and train xgboost and train model
    """
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    n_cats = len(unique(y))
    params = {
        'max_depth': 7,
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': n_cats,
        'eta': 0.3,  # the training step for each iteration
        # 'n_gpus': 0
    }
    if use_gpu:
        params["device"] = "cuda"
    
    # # Set up parameters for xgboost
    # param = {
    # 'max_depth': 6,  # the maximum depth of each tree
    # 'eta': 0.3,  # the training step for each iteration
    # 'objective': 'multi:softprob',  # error evaluation for multiclass training
    # 'num_class': len(np.unique(y_train_full_encoded)) } # the number of classes

    # evallist = [(dval, 'eval'), (dtrain, 'train')]
    # bst = xgb.train(param, dtrain, num_round, evallist)
    bst = xgb.train(params, dtrain, num_round, evals=[(dvalid, 'valid')], early_stopping_rounds=10, verbose_eval=10)
    return bst

def load_xgboost(model_path: Path|str, use_gpu:bool=False) -> xgb.Booster|None:
    """
    Load an XGBoost classifier model from a file.

    Parameters:
    model_path (str): The file path to the saved XGBoost model.
    use_gpu (bool): Flag to indicate whether to use GPU for model loading and prediction.

    Returns:
    model: The loaded XGBoost classifier model, or None if loading fails.
    """

    # Check if the file exists
    model_path = Path(model_path) if isinstance(model_path, str) else model_path

    if not model_path.exists():
        print(f"Error: The file '{model_path}' does not exist.")
        return None

    # Set the global configuration for XGBoost
    try:
        # # Load the model
        # if use_gpu:
        #     model = xgb.Booster(predictor="gpu_predictor")
        # else:
        #     model = xgb.XGBClassifier(predictor="cpu_predictor")
        model = xgb.Booster()
        model.load_model(model_path)

        return model
    except Exception as e:
        # Handle exceptions that occur during model loading
        print(f"Failed to load the model: {e}")
        return None



def test_xgboost(bst:xgb.Booster, 
    adata:AnnData, 
    label_encoder:LabelEncoder, 
    label_key:str='cell_type'
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
    report = classification_report(y_test, best_preds, target_names=classes, output_dict=True)

    # Convert predictions to DataFrame
    preds_df = DataFrame(preds, columns=classes,index=index)

    # Convert the predictions into class labels
    # there's a pandas argmax way thats cleaner
    best_preds = asarray([argmax(line) for line in preds])
    # Add actual and predicted labels to DataFrame
    preds_df['label'] = label_encoder.inverse_transform(best_preds)

    # # Evaluate the model on the test set
    # print(classification_report(y_test, best_preds, target_names=classes))

    return preds_df, report
