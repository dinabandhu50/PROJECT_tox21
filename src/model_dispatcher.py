from sklearn import ensemble
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier


# parameters
rf_param = {
    'max_depth': 11, 
    'n_estimators': 165, 
    'criterion': 'gini'
    }
xgb_param = {
    "verbosity": 0,
    "use_label_encoder": False,
    "booster": "gbtree",
    "tree_method":'gpu_hist', 
    "gpu_id":0,
    'eval_metric': 'logloss', 
    'max_depth': 4, 
    'n_estimators': 100, 
    'gamma': 0.39816492539415976, 
    'reg_alpha': 0.5904253120336852, 
    'reg_lambda': 0.7067282924793774, 
    'learning_rate': 0.2790807703569782
    }
cat_param = {
    "iterations":100,
    "learning_rate":1,
    "task_type":"GPU",
    "devices":'0:1',
    "depth":2,
    "verbose":False
    }

# model dictionary
models = {
    "rf": ensemble.RandomForestClassifier(**rf_param),
    "xgb":MultiOutputClassifier(XGBClassifier(**xgb_param)),
    "cat":CatBoostClassifier(**cat_param),
}