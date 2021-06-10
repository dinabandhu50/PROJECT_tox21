import os

# root path
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# raw data
RAW_TRAIN_FEATURES = os.path.join(ROOT_PATH,'input','raw','tox21_dense_train.csv')
RAW_TRAIN_LABELS = os.path.join(ROOT_PATH,'input','raw','tox21_labels_train.csv')

# processed data
TRAIN_FEATURES = os.path.join(ROOT_PATH,'input','processed','train_features.csv')
TEST_FEATURES = os.path.join(ROOT_PATH,'input','processed','test_features.csv')

TRAIN_TARGETS_OHE = os.path.join(ROOT_PATH,'input','processed','train_targets_ohe.csv')
TRAIN_TARGETS_FOLDS = os.path.join(ROOT_PATH,'input','processed','train_targets_folds.csv')

TEST_TARGETS_OHE = os.path.join(ROOT_PATH,'input','processed','test_targets_ohe.csv')

# saving models and pipelines
MODEL = os.path.join(ROOT_PATH,'models')
PIPE = os.path.join(ROOT_PATH,'models')

# variables

if __name__ == '__main__':
    print(ROOT_PATH)