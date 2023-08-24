import json

import numpy as np

from model.regression import RegressionInterface


class BinaryRegression(RegressionInterface):
    state_file_path_prefix = 'regression'

    def __init__(self, weights, epsilon: float, decay: float, regularizaton: float):
        self.weights = np.array(weights)
        self.epsilon = epsilon
        self.decay = decay
        self.regularizaton = regularizaton

    @classmethod
    def get_context_file_name(cls, model_name) -> str:
        return f'{cls.state_file_path_prefix}_{model_name}.json'

    def save_state(self, model_name: str) -> str:
        file_name = self.get_context_file_name(model_name)
        with open(file_name, mode='w') as f:
            json.dump({
                'weights': list(self.weights),
                'epsilon': self.epsilon,
                'decay': self.decay,
                'regularizaton': self.regularizaton,
            }, f)
        return file_name

    @classmethod
    def load_from_file(cls, model_name: str) -> 'BinaryRegression':
        file_name = cls.get_context_file_name(model_name)
        with open(file_name, mode='r') as f:
            return cls(**json.load(f))

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.weights)

    def loss(self, X: np.ndarray, y: np.ndarray, y_predict: np.ndarray) -> np.ndarray:
        return 2 * (X.transpose().dot(y_predict - y) / X.shape[0] + self.regularizaton * np.linalg.norm(self.weights))

    def update(self, loss):
        self.weights -= self.epsilon * loss
        self.epsilon *= self.decay

    def train_one_batch(self, X: np.ndarray, y: np.ndarray):
        y_predict = self.forward(X)
        l = self.loss(X, y, y_predict)
        self.update(l)

    def train(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 1, batch_size: int = None) -> None:
        for _ in range(epochs):
            for b in range(0, X.shape[0], batch_size):
                X_batch = X[b: min(b + batch_size, X.shape[0])]
                y_batch = y[b: min(b + batch_size, X.shape[0])]
                self.train_one_batch(X_batch, y_batch)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_predict = self.forward(X)

        def func(a):
            if a > .5:
                return 1.0
            return 0.0

        return np.vectorize(func)(y_predict)
