from torch.utils.data import Dataset, SubsetRandomSampler
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from typing import List
import numpy as np
from PIL import Image

class IndexedDataset(Dataset):
    def __init__(self, dir_path, transform=None, test=False, data_filter=False):
        self.dir_path = dir_path
        self.transform = transform
        self.__filter = data_filter

        self.image_filenames = list(Path(self.dir_path).rglob("*.jpg"))

        if test:
            self.labels: List[int] = [
                int(f.parent.stem.split("_")[0]) for f in self.image_filenames
            ]
            self.unlabeled_mask = np.zeros(len(self.image_filenames))
        else:
            self.labels = [-1] * len(self.image_filenames)
            self.unlabeled_mask = np.ones(len(self.image_filenames))

    @property
    def filter():
        return self.__filter

    @filter.setter
    def filter(self, value):
        self.__filter = value

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img = Image.open(self.image_filenames[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        lbl_idx = self.labels[idx]
        if self.__filter:
            assert lbl_idx > -1
        return img, lbl_idx, idx

    def display(self, idx):
        img_name = self.image_filenames[idx]
        print(img_name)
        plt.imshow(mpimg.imread(img_name))
        plt.show()

    def update_label(self, idx, new_label):
        self.labels[idx] = new_label
        self.unlabeled_mask[idx] = 0

    def label_from_filename(self, idx):
        self.labels[idx] = int(self.image_filenames[idx].parent.stem[0])
        self.unlabeled_mask[idx] = 0
