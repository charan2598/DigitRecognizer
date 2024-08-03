# This dataloader is to load pixel data from a csv file.
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class DigitImageDataset(Dataset):
    def __init__(
        self, csv_file, split=0.8, validation=False, testset=False, transform=None, shuffle=True
    ):
        self.testset = testset
        self.validation = validation
        if shuffle:
            data = pd.read_csv(csv_file).sample(frac=1)
        else:
            data = pd.read_csv(csv_file)
        
        data = data.to_numpy(dtype=np.uint8)

        if not self.testset:
            split_index = int(split * data.shape[0])
            if self.validation:
                self.labels, self.images = (
                    data[split_index:, 0],
                    data[split_index:, 1:],
                )
            else:
                self.labels, self.images = (
                    data[:split_index, 0],
                    data[:split_index, 1:],
                )
        else:
            self.images = data

        self.images = self.images.reshape(-1, 28, 28)
        if transform is None:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
            )
        else:
            self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        if not self.testset:
            label = torch.tensor(self.labels[index])

        image = self.transform(self.images[index, :, :])

        if not self.testset:
            return label, image
        else:
            return image


if __name__ == "__main__":
    import os

    csv_file = os.path.join("data", "train.csv")
    dataset = DigitImageDataset(csv_file=csv_file)
    print("Length of the dataset: ", len(dataset))
    data = dataset[0]
    print(
        f"Dimension of the elements returned by dataset: [0]->{data[0].shape}, [1]->{data[1].shape}"
    )
    print("Label: ", data[0])
    print("Image: ", data[1])
    print("Image.dtype: ", data[1].dtype)
