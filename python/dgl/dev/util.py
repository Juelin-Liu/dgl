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
    def __init__(self, target_idx: Tensor, batch_size: int, shuffle: bool, drop_last: bool = True):
        assert (target_idx.device != device("cpu"))
        self.nids = target_idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.cur_idx = 0

    def __iter__(self):
        self.cur_idx = 0
        if self.shuffle:
            print("start shuffling")
            idx = randperm(self.nids.shape[0])
            self.nids = self.nids[idx].clone()
        return self

    def __next__(self):
        if self.drop_last:
            if self.cur_idx + self.batch_size < self.nids.shape[0]:
                seeds = self.nids[self.cur_idx: self.cur_idx + self.batch_size]
                self.cur_idx += self.batch_size
                return seeds
            else:
                raise StopIteration
        elif self.cur_idx < self.nids.shape[0] - 1:
            seeds = self.nids[self.cur_idx: self.cur_idx + self.batch_size]
            self.cur_idx += self.batch_size
            return seeds
        else:
            raise StopIteration
