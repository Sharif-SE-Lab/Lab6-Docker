from abc import ABC, abstractmethod

import numpy as np

from model.feature_cleanser import FeatureCleanser


class DataLoaderInterface(ABC):
    feature_cleanser: 'FeatureCleanser'

    @abstractmethod
    def set_dataset(self, dataset):
        pass

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        pass
