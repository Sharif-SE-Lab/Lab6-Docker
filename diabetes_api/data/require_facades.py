from typing import List

from django_nameko import get_pool


class ModelFacade:
    name = 'model'

    @classmethod
    def get_instance(cls) -> 'ModelFacade':
        with get_pool().next() as pool:
            return getattr(pool, cls.name)

    def train_batch(self, dataset: List[List[float]]):
        pass

    def predict(self, X: List[List[float]]):
        pass


class ModelConfigurationFacade:
    name = 'model_configurations'

    @classmethod
    def get_instance(cls) -> 'ModelConfigurationFacade':
        with get_pool().next() as pool:
            return getattr(pool, cls.name)

    def update_configs(self, **configs):
        pass
