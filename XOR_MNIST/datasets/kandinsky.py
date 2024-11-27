import time

from backbones.disent_encoder_decoder import (
    DecoderConv64,
    EncoderConv64,
)
from backbones.resnet import ResNetEncoder
from datasets.utils.base_dataset import BaseDataset, KAND_get_loader
from datasets.utils.kand_creation import KAND_Dataset


class Kandinsky(BaseDataset):
    NAME = "kandinsky"

    def get_data_loaders(self):
        start = time.time()

        dataset_train = KAND_Dataset(
            base_path="data/kandinsky-patterns-60k",
            split="train",
            preprocess=self.args.preprocess,
            finetuning=self.args.finetuning,
        )
        dataset_val = KAND_Dataset(
            base_path="data/kandinsky-patterns-60k",
            split="val",
            preprocess=self.args.preprocess,
        )
        dataset_test = KAND_Dataset(
            base_path="data/kandinsky-patterns-60k",
            split="test",
            preprocess=self.args.preprocess,
        )
        # dataset_ood   = KAND_Dataset(base_path='data/kandinsky/data',split='ood')

        dataset_train.mask_concepts("red-and-squares")

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(dataset_train),
            "\n val:",
            len(dataset_val),
        )
        print(
            " len test:", len(dataset_test)
        )  # , '\n len ood', len(dataset_ood))

        if not self.args.preprocess:
            train_loader = KAND_get_loader(
                dataset_train, self.args.batch_size, val_test=False
            )
            val_loader = KAND_get_loader(
                dataset_val, 1000, val_test=True
            )
            test_loader = KAND_get_loader(
                dataset_test, 1000, val_test=True
            )
        else:
            train_loader = KAND_get_loader(
                dataset_train, 1, val_test=False
            )
            val_loader = KAND_get_loader(
                dataset_val, 1, val_test=True
            )
            test_loader = KAND_get_loader(
                dataset_test, 1, val_test=True
            )

        # self.ood_loader = get_loader(dataset_ood,  self.args.batch_size, val_test=True)

        return train_loader, val_loader, test_loader

    def get_backbone(self, args=None):
        if self.args.preprocess:
            return ResNetEncoder(
                z_dim=18, z_multiplier=2
            ), DecoderConv64(
                x_shape=(3, 64, 64), z_size=18, z_multiplier=2
            )
        else:
            return EncoderConv64(
                x_shape=(3, 64, 64), z_size=18, z_multiplier=2
            ), DecoderConv64(
                x_shape=(3, 64, 64), z_size=18, z_multiplier=2
            )

    def get_split(self):
        return 3, ()
