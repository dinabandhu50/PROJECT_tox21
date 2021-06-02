import os
import numpy as np
import config
import joblib
import logging
import pandas as pd
from pipeline import pipe0
import model_dispatcher

from sklearn import metrics

def run_training(fold,model_name):
    df_features = pd.read_csv(config.TRAIN_FEATURES)
    df_targets = pd.read_csv(config.TRAIN_LABEL_FOLDS)

    feature_columns = df_features.drop("id",axis=1).columns
    target_columns = df_targets.drop(["id","kfold"],axis=1).columns

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
    # clf = MultiOutputClassifier(model_dispatcher.models[model_name])
    clf = model_dispatcher.models[model_name]

    # fit model on training data
    clf.fit(xtrain, ytrain)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds_proba = clf.predict_proba(xvalid)
    valid_preds_proba = np.array(valid_preds_proba)[:, :, 1].T
    valid_preds = clf.predict(xvalid)

    # get roc auc score
    auc = metrics.roc_auc_score(yvalid, valid_preds_proba)

    # get f1 score
    f1 = metrics.f1_score(yvalid, valid_preds,average="micro")
    
    # log loss
    lloss = metrics.log_loss(np.ravel(yvalid), np.ravel(valid_preds))

    # printing and logging
    print(f'fold={fold}, n_pca={xvalid.shape[1]}, auc={auc:.7}, f1={f1:.7}, logloss={lloss:.7}')
    logging.basicConfig(
        filename=os.path.join(config.ROOT_PATH,"logs",f"{model_name}_base.log"),
        filemode='a',
        level=logging.INFO)
    logging.info(f'fold={fold}, n_pca={xvalid.shape[1]}, auc={auc:.7f}, f1={f1:.7f}, logloss={lloss:.7f}')
    
    # # save the model
    # joblib.dump(pipe0,os.path.join(config.PIPE, f"pipe0_{fold}.bin"))
    joblib.dump(clf,os.path.join(config.MODEL, f"{model_name}_{fold}.bin"))

    # valid_df.loc[:,f"{model_name}_pred"] = valid_preds
    # return valid_df


def model_evaluation(model_name):
    df_features = pd.read_csv(config.TEST_FEATURES)
    df_targets = pd.read_csv(config.TEST_TARGETS)


if __name__ == '__main__':
    model_name = "rf"
    # model_name = "xgb"
    # model_name = "cat"
    for f in range(5):
        run_training(fold=f,model_name=model_name)