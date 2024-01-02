import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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

def train_xgboost(X,y,num_round = 50 ): 
    """
    wrapper to split validation set and train xgboost and train model
    """
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    n_cats = len(np.unique(y))
    params = {
        'max_depth': 7,
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': n_cats,
        'eta': 0.3,  # the training step for each iteration
        # 'n_gpus': 0
    }
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

def test_xgboost(bst, adata, label_encoder, label_key='cell_type'):
    
    X_test = adata.X
    y_test = label_encoder.transform(adata.obs[label_key])

    dtest = xgb.DMatrix(X_test, label=y_test)
    classes = label_encoder.classes_
    # Predict the probabilities for each class on the test set
    preds = bst.predict(dtest)

    # Convert the predictions into class labels
    best_preds = np.asarray([np.argmax(line) for line in preds])

    # Evaluate the model on the test set
    print(classification_report(y_test, best_preds, target_names=classes))

