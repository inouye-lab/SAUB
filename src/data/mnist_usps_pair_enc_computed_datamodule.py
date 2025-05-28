from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, USPS
from torchvision.transforms import v2

import numpy as np
import os

class PairedImageDataset(Dataset):
    def __init__(self, dataset1:Dataset, dataset2:Dataset, counter:int):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # print(f"Reloading with random seed: {counter}!!!!!!!!!!!")

        np.random.seed(counter)

        # # Determine which dataset is smaller
        self.dataset_size = max(len(dataset1), len(dataset2))
        self.rand_idx_1 = np.random.permutation(len(dataset1))
        self.rand_idx_2 = np.random.permutation(len(dataset2))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Sample from both datasets using the smaller dataset's index
        if len(self.dataset1) < len(self.dataset2):
            dataset1 = self.dataset1[self.rand_idx_1[idx % len(self.dataset1)]]
            dataset2 = self.dataset2[self.rand_idx_2[idx]]
        else:
            dataset1 = self.dataset1[self.rand_idx_1[idx]]
            dataset2 = self.dataset2[self.rand_idx_2[idx % len(self.dataset2)]]

        return idx, dataset1, dataset2


class EncodedMNISTUSPS(Dataset):
    def __init__(self, loaded_dataset):
        images_batch, encoded_vectors, targets = loaded_dataset
        self.images_batch = images_batch
        self.encoded_vectors = encoded_vectors
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images_batch[idx], self.encoded_vectors[idx], self.targets[idx]

class MNISTUSPSPairEncModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_split: Tuple[float, float] = (0.9, 0.1),
        postfix: str = None,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transform = v2.Compose([
        #     v2.ToImage(),
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.Resize(size=(32, 32))
        # ])

        self.train_set_mnist: Optional[Dataset] = None
        self.val_set_mnist: Optional[Dataset] = None

        self.train_set_usps: Optional[Dataset] = None
        self.val_set_usps: Optional[Dataset] = None

        self.test_set_mnist: Optional[Dataset] = None
        self.test_set_usps: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        # make sure each epoch is getting different pair
        self.counter = 0


    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)
        # USPS(self.hparams.data_dir, train=True, download=True)
        # USPS(self.hparams.data_dir, train=False, download=True)
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.test_set_mnist and not self.train_set_mnist and not self.val_set_mnist:

            train_val_set_mnist = EncodedMNISTUSPS(torch.load(
                os.path.join(self.hparams.data_dir,
                             f'encoded_mnist/encoded_mnist_train_{self.hparams.postfix}.pth'))
            )
            train_val_set_usps = EncodedMNISTUSPS(torch.load(
                os.path.join(self.hparams.data_dir,
                             f'encoded_usps/encoded_usps_train_{self.hparams.postfix}.pth'))
            )

            self.test_set_mnist = EncodedMNISTUSPS(torch.load(
                os.path.join(self.hparams.data_dir,
                             f'encoded_mnist/encoded_mnist_test_{self.hparams.postfix}.pth'))
            )
            self.test_set_usps = EncodedMNISTUSPS(torch.load(
                os.path.join(self.hparams.data_dir,
                             f'encoded_usps/encoded_usps_test_{self.hparams.postfix}.pth'))
            )

            self.train_set_mnist, self.val_set_mnist = random_split(
                dataset=train_val_set_mnist,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.train_set_usps, self.val_set_usps = random_split(
                dataset=train_val_set_usps,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        self.counter += 1
        train_set_paired = PairedImageDataset(self.train_set_mnist, self.train_set_usps, self.counter)
        return DataLoader(
            dataset=train_set_paired,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        val_set_paired = PairedImageDataset(self.val_set_mnist, self.val_set_usps, 0)
        return DataLoader(
            dataset=val_set_paired,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        train_set_paired = PairedImageDataset(self.test_set_mnist, self.test_set_usps, 0)
        return DataLoader(
            dataset=train_set_paired,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = MNISTUSPSPairEncModule()
