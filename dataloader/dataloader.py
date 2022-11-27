from torch.utils.data import DataLoader
from dataloader.datasets import ClassificationDataset
import torch


def get_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
