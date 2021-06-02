import os

# paths
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_TRAIN_FEATURES = os.path.join(ROOT_PATH,'input','raw','tox21_dense_train.csv')
RAW_TRAIN_LABELS = os.path.join(ROOT_PATH,'input','raw','tox21_labels_train.csv')

PROCESSED_TRAIN_LABELS = os.path.join(ROOT_PATH,'input','processed','targets_ohe.csv')
TRAIN_LABEL_FOLDS = os.path.join(ROOT_PATH,'input','processed','targets_folds.csv')

TRAIN_FEATURES = os.path.join(ROOT_PATH,'input','processed','features.csv')

MODEL = os.path.join(ROOT_PATH,'models')
PIPE = os.path.join(ROOT_PATH,'models')

# variables

if __name__ == '__main__':
    print(ROOT_PATH) 