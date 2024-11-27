import time

from backbones.kand_encoder import TripleCNNEncoder, TripleMLP
from datasets.utils.base_dataset import BaseDataset, KAND_get_loader
from datasets.utils.kand_creation import (
    KAND_Dataset,
    miniKAND_Dataset,
)


class MiniKandinsky(BaseDataset):
    NAME = "minikandinsky"

    def get_data_loaders(self):
        start = time.time()

        if not hasattr(self, "dataset_train"):
            self.dataset_train = miniKAND_Dataset(
                base_path="data/kand-3k",
                split="train",
                finetuning=self.args.finetuning,
            )

        dataset_val = miniKAND_Dataset(
            base_path="data/kand-3k", split="val"
        )
        dataset_test = miniKAND_Dataset(
            base_path="data/kand-3k", split="test"
        )
        # dataset_ood   = KAND_Dataset(base_path='data/kandinsky/data',split='ood')

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(dataset_val),
        )
        print(
            " len test:", len(dataset_test)
        )  # , '\n len ood', len(dataset_ood))

        if not self.args.preprocess:
            train_loader = KAND_get_loader(
                self.dataset_train,
                self.args.batch_size,
                val_test=False,
            )
            val_loader = KAND_get_loader(
                dataset_val, 500, val_test=True
            )
            test_loader = KAND_get_loader(
                dataset_test, 500, val_test=True
            )
        else:
            train_loader = KAND_get_loader(
                self.dataset_train, 1, val_test=False
            )
            val_loader = KAND_get_loader(
                dataset_val, 1, val_test=True
            )
            test_loader = KAND_get_loader(
                dataset_test, 1, val_test=True
            )

        # self.ood_loader = get_loader(dataset_ood,  self.args.batch_size, val_test=True)

        return train_loader, val_loader, test_loader

    def give_full_supervision(self):
        if not hasattr(self, "dataset_train"):
            self.dataset_train = miniKAND_Dataset(
                base_path="data/kand-3k",
                split="train",
                finetuning=self.args.finetuning,
            )
        self.dataset_train.concepts = (
            self.dataset_train.original_concepts
        )

    def give_supervision_to(self, data_idx, figure_idx, obj_idx):
        if not hasattr(self, "dataset_train"):
            self.dataset_train = miniKAND_Dataset(
                base_path="data/kand-3k",
                split="train",
                finetuning=self.args.finetuning,
            )
        self.dataset_train.concepts = (
            self.dataset_train.original_concepts
        )
        self.dataset_train.mask_concepts_specific(
            data_idx, figure_idx, obj_idx
        )

    def get_train_loader_as_val(self):
        return KAND_get_loader(
            self.dataset_train, self.args.batch_size, val_test=True
        )

    def get_backbone(self, args=None):
        return TripleMLP(latent_dim=6), 0
        # return TripleCNNEncoder(latent_dim=6), 0

    def get_split(self):
        return 3, ()
