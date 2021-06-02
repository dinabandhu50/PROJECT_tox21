import os
import config
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


if __name__ == '__main__':
    df = pd.read_csv(config.PROCESSED_TRAIN_LABELS)
    df.loc[:, "kfold"] = -1
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    targets = df.drop("id", axis=1).values

    mskf = MultilabelStratifiedKFold(n_splits=5)

    for fold,(trn, val) in enumerate(mskf.split(X=df,y=targets)):
        df.loc[val,"kfold"] = fold

    df.to_csv(config.TRAIN_LABEL_FOLDS, index=False)
