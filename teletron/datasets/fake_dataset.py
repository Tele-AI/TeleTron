# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.

import torch
import random
import string
from teletron.utils import get_args

class FakeDataset():
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        self.args = get_args()
        self.dst_num_frames = self.args.num_frames
        self.dst_size = tuple(self.args.video_resolution)

    def __len__(self):
        return 10000000

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        random_data = {}
        random_data["struct_prompt"] = ''.join(random.choices(string.ascii_letters + string.digits, k=880))
        random_data["short_prompt"] = ''.join(random.choices(string.ascii_letters + string.digits, k=880))
        random_data["dense_prompt"] = ''.join(random.choices(string.ascii_letters + string.digits, k=880))
        random_data["images"] = torch.randn((self.dst_num_frames, 3, self.dst_size[1], self.dst_size[0]))
        random_data["raw_first_image"] = torch.randn(( 1, 3, self.dst_size[1], self.dst_size[0]))
        return random_data