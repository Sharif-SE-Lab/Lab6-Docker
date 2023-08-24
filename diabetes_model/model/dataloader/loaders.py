from abc import abstractmethod, ABC
from typing import Generic, TypeVar

import numpy as np
import pandas as pd

from model.dataloader import DataLoaderInterface
from model.feature_cleanser import FeatureCleanser

T = TypeVar('T')


class BaseDataloader(Generic[T], DataLoaderInterface, ABC):
    def __init__(self, feature_cleanser: 'FeatureCleanser'):
        self.feature_cleanser = feature_cleanser
        self.dataset = None
        self._X = None
        self._y = None

    def set_dataset(self, dataset: T):
        self.dataset = dataset

    @abstractmethod
    def extract_X(self) -> np.ndarray:
        pass

    @property
    def X(self) -> np.ndarray:
        if self._X is None:
            self._X = self.feature_cleanser.cleanse(self.extract_X())
        return self._X

    @abstractmethod
    def extract_y(self) -> np.ndarray:
        pass

    @property
    def y(self) -> np.ndarray:
        if self._y is None:
            self._y = self.extract_y()
        return self._y


class NumpyDataLoader(BaseDataloader[np.ndarray]):
    def extract_X(self) -> np.ndarray:
        return self.dataset[:, :-1]

    def extract_y(self) -> np.ndarray:
        return self.dataset[:, -1]


class DataframeDataLoader(BaseDataloader[pd.DataFrame]):
    def __init__(self, feature_cleanser: 'FeatureCleanser', label: str):
        super().__init__(feature_cleanser)
        self.label = label

    def extract_X(self) -> np.ndarray:
        return self.dataset.loc[:, self.dataset.columns != self.label].to_numpy()

    def extract_y(self) -> np.ndarray:
        return self.dataset[self.label].to_numpy()


if __name__ == '__main__':
    loader = NumpyDataLoader(np.array([[1, 2, 3, 4, 1], [1, 1, 1, 1, 0]]))
    print(loader.X)
    df = pd.read_csv('diabetes.csv')
    # df = transform(df, label=ifelse(label == "tested_positive", 1, 0))
    df['label'] = np.where(df['label'] == 'tested_positive', 1, 0)
    loader = DataframeDataLoader(df, 'label')
    print(loader.X.shape)
    print(loader.y.shape)
