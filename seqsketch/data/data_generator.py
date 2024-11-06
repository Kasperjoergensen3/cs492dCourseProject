#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import ndjson
import pytorch_lightning as pl
from PIL import Image
from torchvision.transforms import Grayscale, ToTensor, Compose
from torchvision.transforms.functional import invert
from PIL import ImageDraw, Image


class ImageDataset(Dataset):
    def __init__(self, data_list, transform=None, data_size=(256, 256)):
        self.data_list = data_list
        self.transform = transform
        self.data_size = data_size

    def draw_strokes(self, strokes):
        image = Image.new("RGB", (256, 256), "white")
        image_draw = ImageDraw.Draw(image)
        for stroke in strokes:
            points = list(zip(stroke[0], stroke[1]))
            image_draw.line(points, fill=0)
        if self.data_size != (256, 256):
            image = image.resize(self.data_size)
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        sample["next_stroke"] = self.draw_strokes(sample["next_stroke"])
        sample["current_strokes"] = self.draw_strokes(sample["current_strokes"])
        return sample


class QuickDrawDataModule(pl.LightningDataModule):

    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.data_dir = Path(params.data_dir).joinpath(params.category)
        self.category = params.category
        self.val_size = params.val_size
        self.batch_size = params.batch_size
        self.data_size = params.data_size
        self.max_samples = params.max_samples

    def make_data_splitting(self):
        data_split_json = self.data_dir.joinpath("train_test_indices.json")
        with open(data_split_json, "r") as f:
            samples = json.load(f)
        train_samples, val_samples = train_test_split(
            samples["train"], test_size=self.val_size, random_state=42
        )
        self.sample_indices = {
            "train": train_samples,
            "val": val_samples,
            "test": samples["test"],
        }

    def prepare_samples(self, mode="train", seed=42):
        # prepare samples for the given mode
        data_file = self.data_dir.joinpath(f"{self.category}.ndjson")
        with open(data_file, "r") as f:
            data = ndjson.load(f)
        samples = [data[idx] for idx in self.sample_indices[mode]]
        prepared_samples = []
        for sample in samples:
            strokes = sample["drawing"]
            n_strokes = len(strokes)
            for i in range(0, n_strokes):
                sample = {
                    "n_strokes": n_strokes,
                    "step": i,
                    "next_stroke": strokes[i : i + 1],
                    "current_strokes": strokes[:i],
                }
                prepared_samples.append(sample)
        # shuffle the samples
        random.seed(seed)
        random.shuffle(prepared_samples)
        return prepared_samples

    def prepare_data(self):
        if self.data_dir.joinpath(f"datasplit.json").exists():
            with open(self.data_dir.joinpath(f"datasplit.json"), "r") as f:
                samples = json.load(f)
            self.train_samples = samples["train"]
            self.val_samples = samples["val"]
        else:
            self.make_data_splitting()
            self.train_samples = self.prepare_samples("train", seed=42)
            self.val_samples = self.prepare_samples("val", seed=43)
            with open(self.data_dir.joinpath(f"datasplit.json"), "w") as f:
                json.dump({"train": self.train_samples, "val": self.val_samples}, f)
        if self.max_samples:
            self.train_samples = self.train_samples[: self.max_samples]
            self.val_samples = self.val_samples[: self.max_samples]

    def get_transforms(self):
        transforms = Compose(
            [
                Grayscale(num_output_channels=1),
                invert,
                ToTensor(),
            ]
        )
        return transforms

    def setup(self, stage=None):
        # setup for trainer.fit()
        if stage in (None, "fit"):
            self.train_set = ImageDataset(
                self.train_samples, self.get_transforms(), self.data_size
            )
            self.val_set = ImageDataset(
                self.val_samples, self.get_transforms(), self.data_size
            )

        # setup for trainer.test()
        if stage == "test":
            self.test_set = ImageDataset(self.train_samples)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size)
