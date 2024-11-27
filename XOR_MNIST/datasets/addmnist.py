from backbones.addmnist_joint import (
    MNISTPairsDecoder,
    MNISTPairsEncoder,
)
from backbones.addmnist_repeated import MNISTRepeatedEncoder
from backbones.addmnist_single import MNISTSingleEncoder
from datasets.utils.base_dataset import BaseDataset, get_loader
from datasets.utils.mnist_creation import load_2MNIST


class ADDMNIST(BaseDataset):
    NAME = "addmnist"
    DATADIR = "data/raw"

    def get_data_loaders(self):
        dataset_train, dataset_val, dataset_test = load_2MNIST(
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
            args=self.args,
        )

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test

        self.train_loader = get_loader(
            dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = get_loader(
            dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = get_loader(
            dataset_test, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.joint:
            if not self.args.splitted:
                return MNISTPairsEncoder(), MNISTPairsDecoder()
            else:
                return MNISTRepeatedEncoder(), MNISTPairsDecoder()
        else:
            return MNISTSingleEncoder(), MNISTPairsDecoder()

    def get_split(self):
        if self.args.joint:
            return 1, (10, 10)
        else:
            return 2, (10,)

    def get_concept_labels(self):
        return [str(i) for i in range(10)]
