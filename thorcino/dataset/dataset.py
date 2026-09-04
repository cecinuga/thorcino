from pathlib import Path
import random
import numpy as np
from typing import override
from thorcino.dataset.utils import load_jpeg
from thorcino.tensor import Tensor
from abc import ABC, abstractmethod
from collections.abc import Iterator

class Dataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        """Return the sample at the given index."""
        pass

class TensorDataset(Dataset):
    """In-memory dataset over tensors sharing the same leading dimension; indexing
    returns one slice per tensor."""

    def __init__(self, *tensors: Tensor|np.ndarray):
        # Accept raw arrays too: wrapping here keeps `.data` a real ndarray.
        # (`np.ndarray.data` is a memoryview, which can't be indexed per-sample.)
        wrapped = tuple(t if isinstance(t, Tensor) else Tensor(t) for t in tensors)

        # Validate all tensor have same size in dim 0
        first_size = len(wrapped[0].data)
        assert all(len(t.data) == first_size for t in wrapped)
        self.tensors:tuple[Tensor, ...] = wrapped

    @override
    def __len__(self) -> int:
        return len(self.tensors[0])

    @override
    def __getitem__(self, idx: int):
        # Return tuple of slices wrapped in Tensor
        return tuple(Tensor(t.data[idx]) for t in self.tensors)


class ImageDataset(Dataset):
    """Dataset of labelled image paths, decoding each file lazily on access."""

    def __init__(self, image_paths: list[str|Path], labels):
        self.image_paths: list[str|Path] = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # Load image only when requested
        image = load_jpeg(self.image_paths[idx])
        return Tensor(image), Tensor(self.labels[idx])

class DataLoader:
    """Iterates a dataset in batches, stacking each sample field on a new leading axis.
    The final batch is short when the dataset size isn't a multiple of `batch_size`."""

    def __init__(self, dataset: Dataset|list[Tensor], batch_size: int, shuffle: bool = False):
        if isinstance(dataset, list):
            self.dataset:Dataset = Dataset(dataset)
        else:
            self.dataset:Dataset = dataset

        self.batch_size:int = batch_size
        self.shuffle:bool = shuffle

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset)/self.batch_size))

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        # Chunk the indices list
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self._collate_batch(batch)        

    def _collate_batch(self, batch: list[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        num_tensors = len(batch[0]) # e.g., 2 for (features, labels)
        batched_tensors: list[Tensor] = []

        for i in range(num_tensors):
            tensor_list = [sample[i].data for sample in batch]
            stacked = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(stacked))

        return tuple(batched_tensors)