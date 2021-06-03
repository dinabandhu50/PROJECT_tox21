import os
import config
import joblib
import optuna
import logging
import pandas as pd
import numpy as np
from pipeline import pipe0
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn import metrics


def run_training(fold,model_name,param):
    df_features = pd.read_csv(config.TRAIN_FEATURES)
    df_targets = pd.read_csv(config.TRAIN_LABEL_FOLDS)

    feature_columns = df_features.drop("id",axis=1).columns.tolist()
    target_columns = df_targets.drop(["id","kfold"],axis=1).columns.tolist()

    df = pd.merge(df_features,df_targets,how='left',on='id')
    train_df = df[df.kfold !=fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)

    # fit pipe on training + validation features
    full_data = pd.concat([train_df, valid_df],axis=0)

    # train data
    xtrain = train_df[feature_columns]
    ytrain = train_df[target_columns]

    # valid data
    xvalid = valid_df[feature_columns]
    yvalid = valid_df[target_columns]

    # preprocessing
    pipe0.fit(full_data[feature_columns])
    # transform training data
    xtrain = pipe0.transform(xtrain)
    # transform validation data
    xvalid = pipe0.transform(xvalid)

    # initialize classification model
    model = MultiOutputClassifier(XGBClassifier(**param))
    # fit model on training data
    model.fit(xtrain, ytrain)


    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds_proba = model.predict_proba(xvalid)
    valid_preds_proba = np.array(valid_preds_proba)[:, :, 1].T
    valid_preds = model.predict(xvalid)

    # get roc auc score
    auc = metrics.roc_auc_score(yvalid, valid_preds_proba)
    # get f1 score
    f1 = metrics.f1_score(yvalid, valid_preds,average='samples')
    # log loss
    lloss = metrics.log_loss(np.ravel(yvalid), np.ravel(valid_preds))
    # jaccard_score
    js = metrics.jaccard_score(yvalid,valid_preds,average='samples')
    # printing and logging
    print(f'fold={fold}, n_pca={xvalid.shape[1]}, auc={auc:.7f}, f1={f1:.7f}, logloss={lloss:.7f}, js={js:.7f}')

    return auc


def objective(trial):

    param = {
        "verbosity": 0,
        "tree_method": 'exact',
        "use_label_encoder": False,
        "booster": "gbtree",
        "tree_method":'gpu_hist', 
        "gpu_id":0,
        "eval_metric":trial.suggest_categorical("eval_metric", ["error", "logloss","auc","aucpr"]),
        "max_depth" : trial.suggest_int("max_depth", 2, 4),
        "n_estimators" : trial.suggest_int("n_estimators", 10, 1000),
        "gamma" : trial.suggest_float("gamma", 1e-8, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0),        
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0),
    }
    all_aucs = []
    for f in range(5):
        temp_auc = run_training(fold=f,model_name="xgb",param=param)
        all_aucs.append(temp_auc)

    return np.mean(all_aucs)


def predict(model_name="mdoel"):
    df_test = pd.read_csv(config.TEST_DATA)
    ss = pd.read_csv(config.SAMPLE_SUB)
    # feature colymns
    features = [
    f for f in df_test.columns if f not in ("ID")
    ]
    
    fold = 0
    pipe0 = joblib.load(os.path.join(config.PIPE, f"pipe_{fold}.bin"))
    model = joblib.load(os.path.join(config.MODEL, f"rf_{fold}.bin"))

    xtest = pipe0.transform(df_test[features])
    ytest = model.predict(xtest)

    ss["Is_Lead"] = ytest
    ss.to_csv(os.path.join(config.SUB,f"submission_{model_name}_{fold}.csv"),index=False)


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("best trials:")
    trial_ = study.best_trial

    print(trial_.values) 
    print(trial_.params) 