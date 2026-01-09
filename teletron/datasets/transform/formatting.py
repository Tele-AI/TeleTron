from typing import Sequence, Union
import numpy as np
import torch
import random
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def is_seq_of(seq, expected_type):
    """
    检查给定的序列是否包含特定类型的元素。

    :param seq: 要检查的序列
    :param expected_type: 期望的元素类型
    :return: 如果所有元素都是指定类型，则返回True，否则返回False
    """
    if not isinstance(seq, (list, tuple)):
        return False
    return all(isinstance(item, expected_type) for item in seq)


class PackInputs:
    def __init__(self, image_keys, embedding_keys=[], mean=0.5, std=0.5, deterministic=False) -> None:
        self.image_keys = image_keys
        self.embedding_keys = embedding_keys
        self.mean = mean
        self.std = std
        self.deterministic = deterministic

    def __call__(self, data_dict):
        data_dict = self.resize_and_crop(data_dict)
        input_dict = dict()
        input_dict["struct_prompt"] = data_dict["struct_prompt"]
        input_dict["short_prompt"] = data_dict["short_prompt"]
        input_dict["dense_prompt"] = data_dict["dense_prompt"]
        input_dict["frame_interval"] = data_dict["frame_interval"]
        for image_key in self.image_keys:
            input_dict[image_key] = (
                (data_dict[image_key] / 255.0) - self.mean
            ) / self.std
        for embed_key in self.embedding_keys:
            input_dict[embed_key] = data_dict[embed_key]
        return input_dict

    def resize_and_crop(self, data_dict):
        new_height, new_width, dst_height, dst_width = self.get_new_height_width(
            data_dict
        )
        if self.deterministic:
            x1 = (new_width - dst_width) // 2
            y1 = (new_height - dst_height) // 2
        else:
            x1 = random.randint(0, new_width - dst_width)
            y1 = random.randint(0, new_height - dst_height)

        for image_key in self.image_keys:
            images = data_dict[image_key]
            images = F.resize(
                images, (new_height, new_width), InterpolationMode.BILINEAR
            )
            images = F.crop(images, y1, x1, dst_height, dst_width)
            data_dict[image_key] = images
        for embedding_key in self.embedding_keys:
            if embedding_key in ['ref_mask']:
                msk = data_dict[embedding_key]
                msk_height = dst_height // 8
                msk_width = dst_width // 8
                msk = F.resize(
                    msk, (msk_height, msk_width), InterpolationMode.NEAREST
                )
                data_dict[embedding_key] = msk
            else:
                # resize and crop images, too
                cond_image = data_dict[embedding_key]
                cond_image = F.resize(
                    cond_image, (new_height, new_width), InterpolationMode.BILINEAR
                )
                cond_image = F.crop(cond_image, y1, x1, dst_height, dst_width)
                data_dict[embedding_key] = cond_image
                
        return data_dict

    def get_new_height_width(self, data_dict):
        height = data_dict["video_height"]
        width = data_dict["video_width"]
        if len(data_dict["video_info"]) == 2:
            dst_width, dst_height = data_dict["video_info"]
        elif len(data_dict["video_info"]) == 3:
            _, dst_width, dst_height = data_dict["video_info"]
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        return new_height, new_width, dst_height, dst_width
