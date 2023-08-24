from abc import ABC, abstractmethod

import numpy as np


class FeatureCleanser(ABC):
    @abstractmethod
    def cleanse(self, X: np.ndarray) -> np.ndarray:
        pass
