from abc import ABC, abstractmethod
from typing import List

import numpy as np


class RegressionTrainerInterface(ABC):

    @abstractmethod
    def train(self, dataset: np.ndarray):
        pass


class RegressionPredictorInterface(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
