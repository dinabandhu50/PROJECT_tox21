import numpy as np
import config
from sklearn.pipeline import Pipeline

from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.encoding import CountFrequencyEncoder

from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.discretisation import ArbitraryDiscretiser

from sklearn import decomposition


pipe0 = Pipeline([
    ('pca', decomposition.PCA(n_components=600)),
])
