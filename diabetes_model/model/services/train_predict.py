from typing import List

import numpy as np

from enums import MODEL_NAME
from model.dataloader import DataLoaderInterface
from model.regression import RegressionInterface
from model.services.interfaces import RegressionTrainerInterface, RegressionPredictorInterface


class RegressionAdapter(RegressionPredictorInterface, RegressionTrainerInterface):

    def __init__(self, regression: "RegressionInterface", dataloader: "DataLoaderInterface", model_name: str):
        self.regression = regression
        self.dataloader = dataloader
        self.model_name = model_name

    def train(self, dataset: np.ndarray):
        self.dataloader.set_dataset(dataset)
        self.regression.train_one_batch(self.dataloader.X, self.dataloader.y)
        self.regression.save_state(model_name=self.model_name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.dataloader.feature_cleanser.cleanse(X)
        return self.regression.predict(X)


def get_regression_trainer() -> RegressionTrainerInterface:
    from model.regression.src import BinaryRegression
    from model.dataloader.loaders import NumpyDataLoader
    from model.feature_cleanser.chains import FeatureLogisticExtender, FeatureNormalizer
    dataloader = NumpyDataLoader(FeatureLogisticExtender(degree=3, successor=FeatureNormalizer()))
    return RegressionAdapter(
        regression=BinaryRegression.load_from_file(MODEL_NAME),
        dataloader=dataloader,
        model_name=MODEL_NAME
    )


def get_regression_predictor() -> RegressionPredictorInterface:
    from model.regression.src import BinaryRegression
    from model.dataloader.loaders import NumpyDataLoader
    from model.feature_cleanser.chains import FeatureLogisticExtender
    dataloader = NumpyDataLoader(FeatureLogisticExtender(degree=3))
    return RegressionAdapter(
        regression=BinaryRegression.load_from_file(MODEL_NAME),
        dataloader=dataloader,
        model_name=MODEL_NAME
    )
