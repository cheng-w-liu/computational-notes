import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from typing import Set, Tuple

class SiameseMNISTDataset(Dataset):

    def __init__(self, mnist_dataset, labels_set: Set = None):
        self.mnist_dataset = mnist_dataset
        self.train = mnist_dataset.train
        self.transform = mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.targets
            self.train_data = self.mnist_dataset.data  # note that .data contains _untransformed_, original data, in tensor type
            self.labels_set = set(self.train_labels.numpy())

            # self.label_to_indices: {label: indices}
            # for each label, map to the list of example indices that
            #  correspond to this label
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.labels_set
            }

        else:  # self.train == False
            assert labels_set
            self.test_labels = self.mnist_dataset.targets
            self.test_data = self.mnist_dataset.data
            self.labels_set = labels_set.copy()
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.labels_set
            }

            # pre-generate fixed pairs for test set
            random_state = np.random.RandomState(42)

            positive_pairs = [
                [i,
                 random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                 1
                 ]
                for i in range(0, len(self.test_data), 2)
            ]

            negative_pairs = [
                [i,
                 random_state.choice(
                     self.label_to_indices[
                         random_state.choice(
                             list(self.labels_set - set([self.test_labels[i].item()]))
                         )
                     ]
                 ),
                 0
                ]
                for i in range(1, len(self.test_data), 2)
            ]

            self.test_pairs = positive_pairs +  negative_pairs


    def __getitem__(self, index) -> Tuple[Tuple[torch.tensor, torch.tensor], int]:

        if self.train:
            target = np.random.choice([0, 1])
            img1 = self.train_data[index]
            label1 = self.train_labels[index].item()
            if target == 1:
                index2 = index
                while index2 == index:
                    index2 = np.random.choice(self.label_to_indices[label1])
            else:
                label2 = np.random.choice(list(self.labels_set - set([label1])))
                index2 = np.random.choice(self.label_to_indices[label2])
            img2 = self.train_data[index2]

        else:  # self.train == False
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform:
            # note that torchvision.transforms.ToTensor() will convert a 2d numpy array of shape (H, W)
            #  to a tensor of shape (1, H, W)
            # so, originally each example in .data is (28, 28) with values between [0, 255]
            # after transform is applied, the transformed image becomes (1, 28, 28) and values normalized to [0, 1]
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target


    def __len__(self):
        return len(self.mnist_dataset)
