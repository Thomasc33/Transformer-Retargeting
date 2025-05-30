"""
Optimized data loading utilities for improved training performance.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import multiprocessing as mp
from functools import partial


class OptimizedDataLoader:
    """
    Optimized data loader with prefetching and memory management.
    """

    @staticmethod
    def create_optimized_loader(dataset, batch_size, shuffle=True, num_workers=None, pin_memory=True):
        """
        Create an optimized data loader with performance enhancements.

        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes (auto-detected if None)
            pin_memory: Whether to pin memory for faster GPU transfer

        Returns:
            Optimized DataLoader
        """
        if num_workers is None:
            # Use optimal number of workers based on CPU count
            num_workers = min(mp.cpu_count(), 4)  # Cap at 4 to avoid overhead and warnings

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
            drop_last=True if shuffle else False,  # Drop last incomplete batch for training
        )


class MemoryEfficientDataset(Dataset):
    """
    Memory-efficient dataset wrapper that loads data on-demand.
    """

    def __init__(self, data_path, transform=None, cache_size=1000):
        """
        Initialize memory-efficient dataset.

        Args:
            data_path: Path to data file
            transform: Optional transform to apply to data
            cache_size: Number of items to keep in memory cache
        """
        self.data_path = data_path
        self.transform = transform
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []

        # Load metadata without loading full data
        self._load_metadata()

    def _load_metadata(self):
        """Load dataset metadata without loading full data."""
        data = torch.load(self.data_path, map_location='cpu')
        if isinstance(data, dict):
            if 'train' in data:
                self.dataset = data['train']
            else:
                self.dataset = data
        else:
            self.dataset = data

        self.length = len(self.dataset.sampled_data) if hasattr(self.dataset, 'sampled_data') else len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Get item with caching for performance."""
        if idx in self.cache:
            return self.cache[idx]

        # Load item
        if hasattr(self.dataset, 'sampled_data'):
            item = self.dataset.sampled_data[idx]
        else:
            item = self.dataset[idx]

        # Apply transform if specified
        if self.transform:
            item = self.transform(item)

        # Add to cache if there's space
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
            self.cache_order.append(idx)
        elif self.cache_size > 0:
            # Remove oldest item from cache
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
            self.cache[idx] = item
            self.cache_order.append(idx)

        return item


def optimize_data_loading(train_dataset, val_dataset, batch_size, distributed=False, rank=0, world_size=1):
    """
    Create optimized data loaders for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        distributed: Whether using distributed training
        rank: Process rank for distributed training
        world_size: Number of processes for distributed training

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Determine optimal number of workers
    num_workers = min(mp.cpu_count() // world_size if distributed else mp.cpu_count(), 4)

    if distributed and world_size > 1:
        # Use DistributedSampler for distributed training
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
        )
    else:
        # Single-process data loading
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
            drop_last=False
        )

    return train_loader, val_loader


def preload_data_to_memory(dataset, max_items=None):
    """
    Preload dataset items to memory for faster access.

    Args:
        dataset: Dataset to preload
        max_items: Maximum number of items to preload (None for all)

    Returns:
        List of preloaded items
    """
    items = []
    max_items = max_items or len(dataset)
    max_items = min(max_items, len(dataset))

    print(f"Preloading {max_items} items to memory...")
    for i in range(max_items):
        items.append(dataset[i])
        if (i + 1) % 1000 == 0:
            print(f"  Preloaded {i + 1}/{max_items} items")

    print(f"Preloading complete: {len(items)} items in memory")
    return items


class PreloadedDataset(Dataset):
    """
    Dataset wrapper for preloaded data items.
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def create_data_prefetcher(loader, device):
    """
    Create a data prefetcher that loads data to GPU asynchronously.

    Args:
        loader: DataLoader to prefetch from
        device: Target device for prefetching

    Returns:
        Generator that yields prefetched batches
    """
    stream = torch.cuda.Stream()

    def prefetch():
        loader_iter = iter(loader)
        try:
            with torch.cuda.stream(stream):
                batch = next(loader_iter)
                # Move batch to device asynchronously
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in batch]
                else:
                    batch = batch.to(device, non_blocking=True)
        except StopIteration:
            batch = None
        return batch, loader_iter

    batch, loader_iter = prefetch()

    while batch is not None:
        torch.cuda.current_stream().wait_stream(stream)
        yield batch

        try:
            with torch.cuda.stream(stream):
                batch = next(loader_iter)
                # Move batch to device asynchronously
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(device, non_blocking=True) if torch.is_tensor(item) else item for item in batch]
                else:
                    batch = batch.to(device, non_blocking=True)
        except StopIteration:
            batch = None


def estimate_memory_usage(dataset, batch_size, sample_size=10):
    """
    Estimate memory usage for a dataset and batch size.

    Args:
        dataset: Dataset to estimate for
        batch_size: Batch size
        sample_size: Number of samples to use for estimation

    Returns:
        Dictionary with memory usage estimates
    """
    sample_items = []
    for i in range(min(sample_size, len(dataset))):
        sample_items.append(dataset[i])

    # Calculate average item size
    total_size = 0
    for item in sample_items:
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                if torch.is_tensor(sub_item):
                    total_size += sub_item.numel() * sub_item.element_size()
        elif torch.is_tensor(item):
            total_size += item.numel() * item.element_size()

    avg_item_size = total_size / len(sample_items)
    batch_memory = avg_item_size * batch_size

    return {
        'avg_item_size_mb': avg_item_size / (1024 * 1024),
        'batch_memory_mb': batch_memory / (1024 * 1024),
        'batch_memory_gb': batch_memory / (1024 * 1024 * 1024),
        'estimated_peak_memory_gb': batch_memory * 3 / (1024 * 1024 * 1024)  # Rough estimate including gradients
    }
