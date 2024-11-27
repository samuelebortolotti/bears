from copy import deepcopy

import numpy as np
from backbones.addmnist_joint import (
    MNISTPairsDecoder,
    MNISTPairsEncoder,
)
from backbones.addmnist_single import MNISTSingleEncoder
from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.mnist_creation import load_2MNIST


class HALFMNIST(BaseDataset):
    NAME = "halfmnist"
    DATADIR = "data/raw"

    def get_data_loaders(self):
        dataset_train, dataset_val, dataset_test = load_2MNIST(
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
            args=self.args,
        )

        ood_test = self.get_ood_test(dataset_test)

        dataset_train, dataset_val, dataset_test = self.filtrate(
            dataset_train, dataset_val, dataset_test
        )

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.ood_test = ood_test

        self.train_loader = get_loader(
            dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = get_loader(
            dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = get_loader(
            dataset_test, self.args.batch_size, val_test=True
        )
        self.ood_loader = get_loader(
            ood_test, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if not self.args.joint:
            return MNISTSingleEncoder(c_dim=5), MNISTPairsDecoder(
                c_dim=10, latent_dim=10
            )
        else:
            return NotImplementedError("Wrong choice")

    def get_split(self):
        if self.args.joint:
            return 1, (5, 5)
        else:
            return 2, (5,)

    def get_concept_labels(self):
        return [str(i) for i in range(5)]

    def filtrate(self, train_dataset, val_dataset, test_dataset):

        train_c_mask1 = (
            (
                (train_dataset.real_concepts[:, 0] == 0)
                & (train_dataset.real_concepts[:, 1] == 0)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 0)
                & (train_dataset.real_concepts[:, 1] == 1)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 2)
                & (train_dataset.real_concepts[:, 1] == 3)
            )
            | (
                (train_dataset.real_concepts[:, 0] == 2)
                & (train_dataset.real_concepts[:, 1] == 4)
            )
        )

        train_c_mask2 = (
            (
                (train_dataset.real_concepts[:, 1] == 0)
                & (train_dataset.real_concepts[:, 0] == 0)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 0)
                & (train_dataset.real_concepts[:, 0] == 1)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 2)
                & (train_dataset.real_concepts[:, 0] == 3)
            )
            | (
                (train_dataset.real_concepts[:, 1] == 2)
                & (train_dataset.real_concepts[:, 0] == 4)
            )
        )

        train_mask = np.logical_or(train_c_mask1, train_c_mask2)

        val_c_mask1 = (
            (
                (val_dataset.real_concepts[:, 0] == 0)
                & (val_dataset.real_concepts[:, 1] == 0)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 0)
                & (val_dataset.real_concepts[:, 1] == 1)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 2)
                & (val_dataset.real_concepts[:, 1] == 3)
            )
            | (
                (val_dataset.real_concepts[:, 0] == 2)
                & (val_dataset.real_concepts[:, 1] == 4)
            )
        )
        val_c_mask2 = (
            (
                (val_dataset.real_concepts[:, 1] == 0)
                & (val_dataset.real_concepts[:, 0] == 0)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 0)
                & (val_dataset.real_concepts[:, 0] == 1)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 2)
                & (val_dataset.real_concepts[:, 0] == 3)
            )
            | (
                (val_dataset.real_concepts[:, 1] == 2)
                & (val_dataset.real_concepts[:, 0] == 4)
            )
        )

        val_mask = np.logical_or(val_c_mask1, val_c_mask2)

        test_c_mask1 = (
            (
                (test_dataset.real_concepts[:, 0] == 0)
                & (test_dataset.real_concepts[:, 1] == 0)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 0)
                & (test_dataset.real_concepts[:, 1] == 1)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 2)
                & (test_dataset.real_concepts[:, 1] == 3)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 2)
                & (test_dataset.real_concepts[:, 1] == 4)
            )
        )
        test_c_mask2 = (
            (
                (test_dataset.real_concepts[:, 1] == 0)
                & (test_dataset.real_concepts[:, 0] == 0)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 0)
                & (test_dataset.real_concepts[:, 0] == 1)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 2)
                & (test_dataset.real_concepts[:, 0] == 3)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 2)
                & (test_dataset.real_concepts[:, 0] == 4)
            )
        )

        test_mask = np.logical_or(test_c_mask1, test_c_mask2)

        train_dataset.data = train_dataset.data[
            train_mask
        ]  # [:2000, :, :]
        val_dataset.data = val_dataset.data[val_mask]
        test_dataset.data = test_dataset.data[test_mask]

        train_dataset.concepts = train_dataset.concepts[
            train_mask
        ]  # [:2000, :]
        val_dataset.concepts = val_dataset.concepts[val_mask]
        test_dataset.concepts = test_dataset.concepts[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[
            train_mask
        ]  # [:2000]
        val_dataset.targets = np.array(val_dataset.targets)[val_mask]
        test_dataset.targets = np.array(test_dataset.targets)[
            test_mask
        ]

        return train_dataset, val_dataset, test_dataset

    def get_ood_test(self, test_dataset):

        ood_test = deepcopy(test_dataset)

        mask_col0 = (test_dataset.real_concepts[:, 0] >= 0) & (
            test_dataset.real_concepts[:, 1] <= 4
        )
        mask_col1 = (test_dataset.real_concepts[:, 1] >= 0) & (
            test_dataset.real_concepts[:, 0] <= 4
        )

        test_c_mask1 = (
            (
                (test_dataset.real_concepts[:, 0] == 0)
                & (test_dataset.real_concepts[:, 1] == 0)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 0)
                & (test_dataset.real_concepts[:, 1] == 1)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 2)
                & (test_dataset.real_concepts[:, 1] == 3)
            )
            | (
                (test_dataset.real_concepts[:, 0] == 2)
                & (test_dataset.real_concepts[:, 1] == 4)
            )
        )
        test_c_mask2 = (
            (
                (test_dataset.real_concepts[:, 1] == 0)
                & (test_dataset.real_concepts[:, 0] == 0)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 0)
                & (test_dataset.real_concepts[:, 0] == 1)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 2)
                & (test_dataset.real_concepts[:, 0] == 3)
            )
            | (
                (test_dataset.real_concepts[:, 1] == 2)
                & (test_dataset.real_concepts[:, 0] == 4)
            )
        )

        test_mask_in_range = np.logical_and(mask_col0, mask_col1)
        test_mask_value = np.logical_and(~test_c_mask1, ~test_c_mask2)

        test_mask = np.logical_and(
            test_mask_in_range, test_mask_value
        )

        ood_test.data = ood_test.data[test_mask]
        ood_test.concepts = ood_test.concepts[test_mask]
        ood_test.targets = np.array(ood_test.targets)[test_mask]

        return ood_test

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train.data))
        print("Validation samples", len(self.dataset_val.data))
        print("Test samples", len(self.dataset_test.data))
        print("Test OOD samples", len(self.ood_test.data))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--c_sup", type=int, default=1)
    args = parser.parse_args()

    dataset = HALFMNIST(args)
    dataset.print_stats()
    dataset.get_data_loaders()
