from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from data.components.datasets import HeightDataset, collent_fn


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_manifest_file: str,
        valid_manifest_file: str,
        test_manifest_file: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """ """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.train_manifest_file = train_manifest_file
        self.valid_manifest_file = valid_manifest_file
        self.test_manifest_file = test_manifest_file

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
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )
        if stage == "fit":
            self.data_train = HeightDataset(self.train_manifest_file)
            self.data_val = HeightDataset(self.valid_manifest_file)
        elif stage == "test":
            self.data_test = HeightDataset(self.test_manifest_file)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collent_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collent_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collent_fn,
        )


if __name__ == "__main__":
    module = DataModule(
        "data/Manifest/test.jsonp",
        "data/Manifest/valid.jsonp",
        "data/Manifest/test.jsonp",
    )
    module.setup("fit")
    for batch in module.train_dataloader():
        print(f"original:{batch[0].shape}")
        print(f"target:{batch[1].shape}")
        print(f"mask:{batch[2].shape}")
        print(f"image:{batch[3].shape}")
        print(f"height:{batch[4]}")
        exit(1)
