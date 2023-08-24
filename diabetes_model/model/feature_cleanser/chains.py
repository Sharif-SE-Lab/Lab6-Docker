import numpy as np
from sklearn.preprocessing import MinMaxScaler

from model.feature_cleanser import FeatureCleanser


class BaseFeatureCleanser(FeatureCleanser):

    def __init__(self, successor: 'BaseFeatureCleanser' = None):
        self.successor = successor

    def cleanse(self, X: np.ndarray) -> np.ndarray:
        if self.successor is not None:
            return self.successor.cleanse(X)
        return X


class FeatureNormalizer(BaseFeatureCleanser):

    def cleanse(self, X: np.ndarray):
        return MinMaxScaler().fit(X).transform(X)


class FeatureLogisticExtender(BaseFeatureCleanser):
    def __init__(self, degree, successor: 'BaseFeatureCleanser' = None):
        assert degree > 1
        super().__init__(successor)
        self.degree = degree

    def cleanse(self, X: np.ndarray) -> np.ndarray:
        X_ = X
        for d in range(2, self.degree + 1):
            X_ = np.concatenate([X_, X ** d], axis=1)
        return X_
