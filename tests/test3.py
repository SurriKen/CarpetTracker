import numpy as np
import torch

from dataset_processing import DatasetProcessing

device = 'cuda:0'


def get_y_batch(label: list, num_labels: int) -> torch.Tensor:
    lbl = []
    for l in label:
        lbl.append(DatasetProcessing.ohe_from_list([l], num_labels))
    if 'cuda' in device:
        return torch.tensor(lbl, dtype=torch.float, device=device)
    else:
        return torch.tensor(lbl, dtype=torch.float)
x = get_y_batch([0,4, 3], 5)
print()