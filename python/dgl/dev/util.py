from torch import randperm, device, Tensor
from dataclasses import dataclass


@dataclass
class SampleConfig:
    rank: int = 0
    world_size: int = 1
    num_partition: int = 1
    batch_size: int = 1024
    replace: bool = False
    mode: str = "uva"  # must be one of ["uva", "gpu"]
    reindex: bool = True  # whether to make the sampled vertex to be 0 indexed
    fanouts: list[int] = None
    drop_last: bool = False

    @staticmethod
    def header():
        return ["rank", "world_size", "num_partition", "batch_size", "replace", "mode", "reindex", "fanouts"]

    def content(self):
        return [self.rank,
                self.world_size,
                self.num_partition,
                self.batch_size,
                self.replace,
                self.mode,
                self.reindex,
                self.fanouts
                ]


class IdxLoader:
    def __init__(self, target_idx: Tensor, batch_size: int, shuffle: bool, max_step_per_epoch: int):
        assert (target_idx.device != device("cpu"))
        self.nids = target_idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_step_per_epoch = max_step_per_epoch
        self.cur_step = 0
        assert(max_step_per_epoch * batch_size <= target_idx.shape[0])

    def __iter__(self):
        self.cur_step = 0
        if self.shuffle:
            idx = randperm(self.nids.shape[0])
            self.nids = self.nids[idx].clone()
        return self

    def __next__(self):
        if self.cur_step < self.max_step_per_epoch:
            seeds = self.nids[self.cur_step * self.batch_size: (self.cur_step + 1) * self.batch_size]
            self.cur_step += 1
            return seeds
        else:
            raise StopIteration