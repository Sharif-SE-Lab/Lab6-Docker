import json
from typing import List

import numpy as np
from nameko.rpc import rpc

from enums import MODEL_NAME
from model.services import get_regression_trainer, get_regression_predictor


class ModelFacade:
    name = 'model'

    @rpc
    def train_batch(self, dataset: List[List[float]]):
        get_regression_trainer().train(np.array(dataset))
        return {
            'status': 'ok'
        }

    @rpc
    def predict(self, X: List[List[float]]):
        result = get_regression_predictor().predict(np.array(X))
        return {
            'status': 'ok',
            'data': list(result)
        }


class ModelConfigurationFacade:
    name = 'model_configurations'

    @rpc
    def update_configs(self, **configs):
        from model.regression.src import BinaryRegression
        with open(BinaryRegression.get_context_file_name(MODEL_NAME), mode='r') as f:
            current_configs = json.load(f)
        if configs.keys() - current_configs.keys():
            return {
                'status': 'failed',
                'message': 'bad data',
            }
        current_configs.update(configs)
        regression = BinaryRegression(**current_configs)
        regression.save_state(MODEL_NAME)
