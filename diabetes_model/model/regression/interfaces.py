from abc import ABC, abstractmethod

import numpy as np


class RegressionInterface(ABC):

    @abstractmethod
    def save_state(self, model_name: str) -> str:
        pass

    @abstractmethod
    def train_one_batch(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 1, batch_size: int = None) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
