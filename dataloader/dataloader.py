from torch.utils.data import DataLoader
from dataloader.datasets import ClassificationDataset
import torch

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)  
# , num_workers=4, pin_memory=True,

def get_loader_seg(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def get_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
