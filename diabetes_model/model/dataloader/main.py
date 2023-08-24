import numpy as np

from model.dataloader import DataLoaderInterface
from model.dataloader.chains import NormalizerDataLoaderDecorator, LogisticExtenderDataLoaderDecorator
from model.dataloader.loaders import NumpyDataLoader


def main():
    loader = NumpyDataLoader(
        np.array(
            [
                [1, 2, 3, 4, 1],
                [3, 1, 4, 1, 0],
                [2, 1, 4, 6, 0],
                [6, 3, 4, 5, 1],
                [2, 1, 9, 1, 0],
                [8, 2, 4, 2, 0],
                [2, 1, 4, 11, 1],
                [3, 3, 4, 2, 0],
            ]
        )
    )
    loader: DataLoaderInterface = NormalizerDataLoaderDecorator(loader)
    loader: DataLoaderInterface = LogisticExtenderDataLoaderDecorator(loader, degree=2)
    print(loader.X)



if __name__ == '__main__':
    main()
