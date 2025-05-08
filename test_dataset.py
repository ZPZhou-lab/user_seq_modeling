from h11 import Data
from regex import D
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed as dist
from datasets import Dataset

def _set_world_size_and_rank(obj, world_size: int, rank: int):
    """
    Set the world size and rank for distributed training.
    """
    if world_size is None:
        obj.world_size = dist.get_world_size() if dist.is_initialized() else 1
    else:
        obj.world_size = world_size
    
    if rank is None:
        obj.rank = dist.get_rank() if dist.is_initialized() else 0
    else:
        obj.rank = rank


class TableReader:
    def __init__(self, n_samples: int=100):
        self.n_samples = n_samples
        self.reader = [{"id": i, "value": i} for i in range(n_samples)]
    
    def to_pandas(self, start: int, limit: int=-1):
        if limit == -1:
            data = self.reader[start:]
        else:
            end = start + limit
            data = self.reader[start:end]
        return pd.DataFrame(data)

    def __len__(self):
        return self.n_samples
    

class MyDataset:
    def __init__(self, 
        n_samples: int=100, 
        shard_size: int=16,
        batch_size: int=4,
        world_size: int=None,
        rank: int=None
    ):
        """
        Parameters
        ----------
        shard_size: int
            The size of each shard at each node, there are `world_size * shard_size` samples in total
            at the same time.
        """
        _set_world_size_and_rank(self, world_size, rank)
        self.shard_size = shard_size
        self.batch_size = batch_size
        # mock data
        self.reader = TableReader(n_samples)
        assert self.shard_size * self.world_size <= n_samples, \
            f"shard_size * world_size = {self.shard_size * self.world_size} should be less than total samples {len(self.reader)}"

        # create shard indices
        self._shards_idx = []
        for shard_s in range(0, n_samples, shard_size * self.world_size):
            shard_e = shard_s + shard_size * self.world_size
            if shard_e <= n_samples:
                start = shard_s + self.rank * shard_size
                limit = (shard_size // batch_size) * batch_size
            else:
                # split the last part into `world_size` shards if remain > batch_size * world_size
                remain = n_samples - shard_s
                global_batch = self.world_size * self.batch_size
                if remain > global_batch:
                    last_shard_size = (remain // global_batch) * global_batch // self.world_size
                    start = shard_s + self.rank * last_shard_size
                    limit = last_shard_size
                else:
                    # remain can not be split into `world_size` chunks to build a batch
                    break
            self._shards_idx.append((start, limit))
        # update the sample_size
        self._total = sum([limit for _, limit in self._shards_idx]) * self.world_size

        self.current_shard_idx = 0
        self.current_shard = self.safe_load()
    
    def __len__(self):
        return self._total

    def __getitem__(self, global_idx):
        # 处理批量索引的情况
        if isinstance(global_idx, list):
            print("get batchs")
            batchs = [self.__getitem__(idx) for idx in global_idx]
            return batchs
        
        shard_end = self.shard_range[0] + self.shard_range[1]
        # fetch the next shard if the current shard reaches the end
        if global_idx >= shard_end:
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self._shards_idx)
            self.current_shard = self.safe_load()
        elif global_idx < self.shard_range[0]:
            self.current_shard_idx = 0
            self.current_shard = self.safe_load()
        
        # get the local index in the current shard
        local_idx = global_idx - self.shard_range[0]
        return self.current_shard[local_idx]
    
    def safe_load(self, path: str=None):
        """
        Load the dataset from the given path.
        """
        start, limit = self._shards_idx[self.current_shard_idx]
        df = self.reader.to_pandas(start=start, limit=limit)
        return Dataset.from_pandas(df)
    
    @property
    def shard_range(self):
        return self._shards_idx[self.current_shard_idx]
    

class SequentialDistributedSampler(Sampler):
    def __init__(self, dataset: MyDataset, world_size: int=None, rank: int=None):
        self.dataset = dataset
        _set_world_size_and_rank(self, world_size, rank)

    def __iter__(self):
        """
        each node load a shard of data with `shard_size` samples, 
        there are `num_replicas * shard_size` samples in total at the same time.\n
        The sampler samples the sequence of indices from the dataset._shards_idx intervals.
        """
        indices = []
        for start, limit in self.dataset._shards_idx:
            # For each shard assigned to this rank, generate indices within that shard
            for i in range(limit):
                indices.append(start + i)
        return iter(indices)

    def __len__(self):
        shards = [limit for _, limit in self.dataset._shards_idx]
        return sum(shards)

    
def infinite_loader_wrapper(loader):
    """
    Wrap the dataloader to make it infinite.
    """
    while True:
        for batch in loader:
            yield batch


if __name__ == "__main__":
    batch_size = 4
    dataset_0 = MyDataset(n_samples=100, shard_size=20, world_size=4, rank=0, batch_size=batch_size)
    dataset_4 = MyDataset(n_samples=100, shard_size=20, world_size=4, rank=3, batch_size=batch_size)
    sampler_0 = SequentialDistributedSampler(dataset_0, world_size=4, rank=0)
    sampler_4 = SequentialDistributedSampler(dataset_4, world_size=4, rank=3)
    dataloader_0 = DataLoader(dataset_0, sampler=sampler_0, batch_size=batch_size)
    dataloader_4 = DataLoader(dataset_4, sampler=sampler_4, batch_size=batch_size)
    print(f"Total samples: {len(dataset_0)}")
    print(f"Total samples in each shard: {dataset_0._shards_idx}")
    
    loader = infinite_loader_wrapper(dataloader_0)
    for _ in range(10):
        batch = next(loader)
        print(batch)