from abc import abstractmethod
from argparse import Namespace
from typing import List, Tuple

import numpy as np
import torch.optim
from torch import nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms


class BaseDataset:
    """
    Base Dataset for NeSy.
    """

    NAME = None
    DATADIR = None

    # TRANSFORM = None
    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.args = args

    @abstractmethod
    def get_concept_labels(self) -> List[str]:
        """
        Simple abstract method to return the labels of the concepts
        """
        pass

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass


def get_loader(dataset, batch_size, num_workers=4, val_test=False):

    if val_test:
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
    else:

        labels = dataset.targets
        tot = np.max(labels)

        class_sample_count = np.array(
            [len(np.where(labels == t)[0]) for t in np.unique(labels)]
        )

        tot = np.ones(np.max(labels) + 1)
        weight = 1.0 / class_sample_count

        j = 0
        for i in range(np.max(labels) + 1):
            if i in np.unique(labels):
                tot[i] = weight[j]
                j += 1

        samples_weight = np.array([tot[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)

        sampler = WeightedRandomSampler(
            samples_weight.type("torch.DoubleTensor"),
            len(samples_weight),
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
        )


def KAND_get_loader(
    dataset,
    batch_size,
    num_workers=4,
    val_test=False,
    preprocess=False,
):

    if val_test:
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )
    else:
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
