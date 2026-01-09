import torch
import math
import itertools

class BucketSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, global_batch_size, seed=42, drop_last=True, shuffle=True, infinite=True):
        self.dataset = dataset
        self.total_samples = len(dataset)
        self.pre_consumed_samples = consumed_samples
        self.consumed_micro_batch = 0

        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        self.bucket_indices = self.dataset.get_bucket_index_list()

        self.compute_num_samples()

        self.total_size = self.total_micro_batch * self.data_parallel_size * self.micro_batch_size

        self.bucket_weights = torch.tensor(self.dataset.buckets_size_ratio, dtype=torch.float)

        self.bucket_indices_iters = [self.fixed_infinite_bucket_iter(bucket) for bucket in self.bucket_indices]
        
        self.global_batch_size = global_batch_size
        self.iteration = 0

    def fixed_infinite_bucket_iter(self, bucket):
        while True:
            g = torch.Generator().manual_seed(self.seed + self.consumed_micro_batch)
            index = torch.randperm(len(bucket), generator=g).tolist()
            bucket_indices = [bucket[i] for i in index]
            for i in range(0, len(bucket_indices), self.micro_batch_size * self.data_parallel_size):
                start_idx = i + self.data_parallel_rank * self.micro_batch_size
                yield bucket_indices[start_idx: start_idx + self.micro_batch_size]

    def compute_num_samples(self):
        self.total_micro_batch = 0
        for i in range(len(self.bucket_indices)):
            if self.drop_last:
                self.total_micro_batch += len(self.bucket_indices[i]) // self.micro_batch_size // self.data_parallel_size
                self.bucket_indices[i] = self.bucket_indices[i][:len(self.bucket_indices[i]) // self.micro_batch_size // self.data_parallel_size * self.micro_batch_size * self.data_parallel_size]
            else:
                self.total_micro_batch += math.ceil(math.ceil(len(self.bucket_indices[i]) / self.micro_batch_size) / self.data_parallel_size)
                self.bucket_indices[i] += self.bucket_indices[i][:math.ceil(math.ceil(len(self.bucket_indices[i]) / self.micro_batch_size) / self.data_parallel_size) * self.micro_batch_size * self.data_parallel_size - len(self.bucket_indices[i])]

    def skip_consumed_batch(self):
        for _ in range(self.pre_consumed_samples // self.micro_batch_size // self.data_parallel_size):
            which_bucket_idx = torch.multinomial(self.bucket_weights, 1, replacement=True, generator=torch.Generator().manual_seed(self.seed+self.consumed_micro_batch)).item()
            self.consumed_micro_batch += 1
            next(self.bucket_indices_iters[which_bucket_idx])

    def __len__(self):
        return self.total_micro_batch

    def __iter__(self):
        if self.pre_consumed_samples > 0:
            self.skip_consumed_batch()

        while True:
            which_bucket_idx = torch.multinomial(self.bucket_weights, 1, replacement=True, generator=torch.Generator().manual_seed(self.seed+self.iteration)).item()
            idx = next(self.bucket_indices_iters[which_bucket_idx])
            self.consumed_micro_batch += 1
            if self.consumed_micro_batch % self.global_batch_size == 0:
                self.iteration += 1
            yield idx
