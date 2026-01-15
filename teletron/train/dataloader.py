import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.core import mpu
from teletron.utils import (
    get_args,
    print_rank_0,
    set_config
)
from teletron.datasets.build import build_train_valid_test_datasets
from teletron.datasets.samplers import build_sampler

class DataloaderMixin:

    def build_train_valid_test_data_loaders(self,
        is_tp_first=None, dp_rank=None, dp_size=None, train_ds_prev=None, valid_ds_prev=None, return_ds=False
    ):
        args = get_args()

        (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

        print_rank_0("> building train, validation, and test datasets ...")

        if args.iteration > 0 and args.consumed_train_samples == 0:
            assert (
                args.train_samples is None
            ), "only backward compatiblity support for iteration-based training"
            args.consumed_train_samples = args.iteration * args.global_batch_size
        if args.iteration > 0 and args.consumed_valid_samples == 0:
            if args.train_samples is None:
                args.consumed_valid_samples = (
                    (args.iteration // args.eval_interval)
                    * args.eval_iters
                    * args.global_batch_size
                )
                
        if train_ds_prev is not None:
            train_ds = train_ds_prev
            valid_ds = valid_ds_prev
            test_ds = None
        else:
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets()
        train_dataloader = self.build_pretraining_data_loader(
            train_ds, args.consumed_train_samples, dp_rank, dp_size
        )
        if args.skip_train:
            valid_dataloader = self.build_pretraining_data_loader(
                valid_ds, 0, dp_rank, dp_size
            )
        else:
            if valid_ds is None:
                valid_dataloader = None
            else:
                args.consumed_valid_samples = args.consumed_valid_samples % len(valid_ds) if valid_ds is not None else args.consumed_valid_samples
                valid_dataloader = self.build_pretraining_data_loader(
                    valid_ds, args.consumed_valid_samples, dp_rank, dp_size
                )
        test_dataloader = self.build_pretraining_data_loader(
            test_ds, 0, dp_rank, dp_size
        )

        # Flags to know if we need to do training/validation/testing.
        
        do_train =  args.train_iters > 0
        do_valid =  args.eval_iters > 0
        do_test = False
        
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long, device='cuda')

        if dp_rank is None or dp_size is None:
            torch.distributed.broadcast(flags, 0)

        args.do_train = getattr(args, "do_train", False) or flags[0].item()
        args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
        args.do_test = getattr(args, "do_test", False) or flags[2].item()

        if return_ds is True:
            return train_dataloader, valid_dataloader, test_dataloader, train_ds, valid_ds
        else:
            return train_dataloader, valid_dataloader, test_dataloader

    def build_pretraining_data_loader(self, dataset, consumed_samples, data_parallel_rank=None, data_parallel_size=None):
        """Build dataloader given an input dataset."""

        if dataset is None:
            return None
        args = get_args()
        if data_parallel_rank is None:
            data_parallel_rank = mpu.get_data_parallel_rank()
        if data_parallel_size is None:
            data_parallel_size = mpu.get_data_parallel_world_size()

        sampler_config = set_config().get("sampler", None)
        print(sampler_config)
        if sampler_config is None:
            sampler_config = dict(type="DefaultSampler", shuffle=True, seed=42, drop_last=True,infinite=True)
        batch_sampler = build_sampler(sampler_config, 
                                        dataset=dataset, 
                                        consumed_samples=consumed_samples, 
                                        micro_batch_size=args.micro_batch_size, 
                                        data_parallel_rank=data_parallel_rank, 
                                        data_parallel_size=data_parallel_size,
                                        global_batch_size=args.global_batch_size // args.distributed_vae_world_size,
                                    )

        return torch.utils.data.DataLoader(dataset,
                                        batch_sampler=batch_sampler,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        persistent_workers=True if args.num_workers > 0 else False,
                                        )
