import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.io.image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from ..dataset._dataset_info import DatasetWithInfo, DatasetInfo


class PairedClevr(DatasetWithInfo):
    dataset_info = DatasetInfo(
        feature_names=('shape', 'color', 'size', 'material', 'posX', 'posY'),
        feature_counts=(3, 8, 2, 2, 32, 32),
        is_contiguous=(False, False, False, False, True, True),
        n_features=6,
        image_size=(3, 128, 128),
        features_list=[],
        features_range=[],
    )

    def __init__(self, dataset_dir: Path):
        super().__init__(self.dataset_info)

        self.scenes_dir = dataset_dir / 'scenes'
        self.img_dir = dataset_dir / 'images'

        self.img_template = "{name}_{idx:06d}.png"
        self.scene_template = "scene_{name}_{idx:06d}.png"
        self.json_template = "scene_{idx:06d}.json"

        self.scenes_list = sorted(os.listdir(self.scenes_dir))
        self._size = len(self.scenes_list)

    def __len__(self):
        return self._size

    def get_image(self, image_name):
        img_path = os.path.join(self.img_dir, image_name)
        img = read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB) / 255
        return img

    def __getitem__(self, idx):

        # Open json file
        scene = self.scenes_list[idx]
        with open(os.path.join(self.scenes_dir, scene), 'r') as f:
            scenes_list = json.load(f)
        scenes_dict = {row['image_filename']: row for row in scenes_list}

        # Load discrete image, pair and scenes
        image_name = self.img_template.format(name='img', idx=idx)
        image = self.get_image(image_name)
        image_scene = self.get_image(self.scene_template.format(name='img', idx=idx))

        donor_name = self.img_template.format(name='pair', idx=idx)
        donor = self.get_image(donor_name)
        donor_scene = self.get_image(self.scene_template.format(name='pair', idx=idx))

        exchange_labels = self.get_difference(scenes_dict[image_name]['objects'][0],
                                              scenes_dict[donor_name]['objects'][0])

        return (image, donor), (image_scene, donor_scene), exchange_labels

    def get_difference(self, obj1, obj2):
        exchange_labels = torch.zeros(self.dataset_info.n_features, dtype=bool)

        if obj1['shape'] != obj2['shape']:
            exchange_labels[0] = True
        elif obj1['color'] != obj2['color']:
            exchange_labels[1] = True
        elif obj1['size'] != obj2['size']:
            exchange_labels[2] = True
        elif obj1['material'] != obj2['material']:
            exchange_labels[3] = True
        elif obj1['3d_coords'][0] != obj2['3d_coords'][0]:
            exchange_labels[4] = True
        elif obj1['3d_coords'][1] != obj2['3d_coords'][1]:
            exchange_labels[5] = True
        else:
            raise ValueError(f'All features are the same {obj1}, {obj2}')
        return exchange_labels.unsqueeze(-1)


class PairedClevrDatamodule(pl.LightningDataModule):
    dataset_type: DatasetWithInfo = PairedClevr
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, path_to_data_dir: str = '../data/',
                 batch_size: int = 64,
                 mode: str = "paired_clevr",
                 num_workers: int = 4):
        super().__init__()
        self.num_workers = num_workers
        path_to_data_dir = Path(path_to_data_dir)
        self.path_to_paired_clevr_dir = path_to_data_dir / mode
        self.batch_size = batch_size
        self.image_size = (3, 128, 128)

    def setup(self, stage):
        self.train_dataset = PairedClevr(dataset_dir=self.path_to_paired_clevr_dir / 'train')

        self.val_dataset = PairedClevr(dataset_dir=self.path_to_paired_clevr_dir / 'val')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    dataset = PairedClevr(
        dataset_dir=Path('/home/yessense/data/paired_codebook_ae/data/paired_clevr/train/'))
    start_idx = 300
    n_images = 2

    plt.figure(figsize=(30, 20))
    fig, ax = plt.subplots(n_images, 4)
    for i in range(n_images):
        (img, pair), (img_scene, donor_scene), exchange_label = dataset[i + start_idx]
        # print(
        #     *[dataset.dataset_info.feature_names[i] for i, exchange in enumerate(exchange_label) if
        #       exchange])
        ax[i, 0].imshow(img.permute(1, 2, 0))
        ax[i, 1].imshow(pair.permute(1, 2, 0))
        ax[i, 2].imshow(img_scene.permute(1, 2, 0))
        ax[i, 3].imshow(donor_scene.permute(1, 2, 0))
    plt.show()
