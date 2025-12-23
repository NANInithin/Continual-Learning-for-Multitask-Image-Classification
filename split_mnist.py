from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


@dataclass
class TaskSpec:
    digits: Tuple[int, int]  # e.g. (0, 1)


class FilteredMNIST(Dataset):
    def __init__(self, base_ds, digits: Tuple[int, int]):
        self.base_ds = base_ds
        self.d0, self.d1 = digits

        # MNIST targets are in base_ds.targets (torch tensor)
        t = base_ds.targets
        mask = (t == self.d0) | (t == self.d1)
        self.idxs = mask.nonzero(as_tuple=False).squeeze(1)

    def __len__(self):
        return self.idxs.numel()

    def __getitem__(self, i):
        x, y = self.base_ds[int(self.idxs[i])]
        # remap labels to {0,1} inside the task
        y = 0 if y == self.d0 else 1
        return x, y


def get_split_mnist_tasks(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> List[dict]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_base = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    test_base  = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)

    task_specs = [
        TaskSpec((0, 1)),
        TaskSpec((2, 3)),
        TaskSpec((4, 5)),
        TaskSpec((6, 7)),
        TaskSpec((8, 9)),
    ]  # Split MNIST tasks [file:1]

    tasks = []
    for spec in task_specs:
        train_ds = FilteredMNIST(train_base, spec.digits)
        test_ds  = FilteredMNIST(test_base, spec.digits)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

        tasks.append({
            "digits": spec.digits,
            "train_loader": train_loader,
            "test_loader": test_loader,
        })

    return tasks