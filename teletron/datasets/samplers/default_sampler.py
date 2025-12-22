import torch
import math

class DefaultSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, global_batch_size, seed=42, drop_last=True, shuffle=True, infinite=True):
        # Keep a copy of input params for later use.
        self.total_samples = len(dataset)
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.infinite = infinite

        if self.drop_last:
            self.num_samples = self.total_samples // self.micro_batch_size // self.data_parallel_size
        else:
            self.num_samples = math.ceil(math.ceil(self.total_samples / self.micro_batch_size) / self.data_parallel_size)
        
        self.total_size = self.num_samples * self.data_parallel_size * self.micro_batch_size
        self.epoch = self.consumed_samples // self.total_size
        self.global_batch_size = global_batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        while True:
            indices = list(range(self.total_samples))
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                idx = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in idx]
            
            if not self.drop_last:
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]
            else:
                indices = indices[: self.total_size]

            if self.consumed_samples % self.total_size != 0:
                indices = indices[self.consumed_samples % self.total_size:]

            for i in range(0, len(indices), self.micro_batch_size * self.data_parallel_size):
                start_idx = i + self.data_parallel_rank * self.micro_batch_size
                yield indices[start_idx: start_idx + self.micro_batch_size]

            self.epoch += 1
            self.consumed_samples += len(indices)

            if not self.infinite:
                break

