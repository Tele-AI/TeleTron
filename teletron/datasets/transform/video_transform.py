import copy
import math
import random
import json
import scipy
import numpy as np
import torch
import os
from teletron.utils import video_utils
import torch.nn.functional as F
from einops import rearrange
from math import floor, ceil
from func_timeout import func_set_timeout


class MaskGenerator:
    def __init__(self, mask_ratios, min_clear_ratio=0.0, max_clear_ratio=1.0):
        valid_mask_names = [
            "t2v",
            "i2v",
            "clear",
            "transition",
            "continuation",
            "random",
            "f1fn2v"
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        self.mask_ratios = mask_ratios
        self.min_clear_ratio = min_clear_ratio
        self.max_clear_ratio = max_clear_ratio

    def get_mask(self, num_frames, height=None, width=None):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break
        num_select = random.randint(floor(num_frames * self.min_clear_ratio),
                                    ceil(num_frames * self.max_clear_ratio))

        if height is not None and width is not None:
            mask = torch.ones(size=(num_frames, 1, height, width), dtype=torch.float32)
        else:
            mask = torch.ones(num_frames, dtype=torch.float32)

        if num_frames <= 1:
            return mask
        if mask_name == "t2v":
            return mask
        elif mask_name == "i2v":
            mask[0] = 0
        elif mask_name == "clear":
            mask[:] = 0
        elif mask_name == "transition":
            mask[0] = 0
            mask[-1] = 0
        elif mask_name == "continuation":
            mask[:num_select] = 0
        elif mask_name == "random":
            selected_indices = random.sample(range(num_frames), num_select)
            mask[selected_indices] = 0
        elif mask_name == "f1fn2v":
            selected_indices = random.sample(range(num_frames), num_select)
            mask[selected_indices] = 0
            mask[0] = 0
        return mask

class MaskProcesser:
    '''
    modified from open-sora-plan
    https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/utils/mask_utils.py
    '''
    def __init__(self, ae_stride_h=8, ae_stride_w=8, ae_stride_t=4, **kwargs):
        self.ae_stride_h = ae_stride_h
        self.ae_stride_w = ae_stride_w
        self.ae_stride_t = ae_stride_t
    
    def __call__(self, mask):
        T, _, H, W = mask.shape
        new_H, new_W = H // self.ae_stride_h, W // self.ae_stride_w
        mask = rearrange(mask, 't c h w -> (t c) 1 h w')
        mask = F.interpolate(mask, size=(new_H, new_W), mode='bilinear')
        mask = rearrange(mask, '(t c) 1 h w -> t c h w', t=T)
        # align with wan vae
        new_T = (T + 3) // self.ae_stride_t
        mask_first_frame = mask[0:1].repeat(self.ae_stride_t, 1, 1, 1).contiguous() 
        mask = torch.cat([mask_first_frame, mask[1:]], dim=0)
        # if T % 2 == 1:
        #     new_T = T // self.ae_stride_t + 1
        #     mask_first_frame = mask[0:1].repeat(self.ae_stride_t, 1, 1, 1).contiguous() 
        #     mask = torch.cat([mask_first_frame, mask[1:]], dim=0)
        # else:
        #     new_T = T // self.ae_stride_t
        mask = mask.view(new_T, self.ae_stride_t, new_H, new_W).contiguous()
        return mask


class GenerateRefImages:
    def __init__(self, mask_cfg=dict(), min_clear_ratio=0.0, max_clear_ratio=1.0):
        self.mask_generator = MaskGenerator(mask_cfg, min_clear_ratio, max_clear_ratio)

    def __call__(self, data_dict):
        ref_images = copy.deepcopy(data_dict["images"])
        num_frames = ref_images.shape[0]
        mask = self.mask_generator.get_mask(num_frames)[:, None, None, None]
        ref_images = ref_images * (mask < 0.5)
        data_dict["ref_images"] = ref_images
        return data_dict

class GenerateRefImagesWithMask:
    def __init__(self, mask_cfg=dict(), min_clear_ratio=0.0, max_clear_ratio=1.0):
        self.mask_generator = MaskGenerator(mask_cfg, min_clear_ratio, max_clear_ratio)
        self.mask_processer = MaskProcesser()

    def __call__(self, data_dict):
        ref_images = copy.deepcopy(data_dict["images"])
        num_frames, height, width = ref_images.shape[0], ref_images.shape[-2], ref_images.shape[-1]
        mask = self.mask_generator.get_mask(num_frames, height=height, width=width)
        ref_images = ref_images * (mask < 0.5)
        data_dict["ref_mask"] = self.mask_processer((mask < 0.5).float())
        data_dict["ref_images"] = ref_images
        return data_dict
    
class GenerateRefImagesWithTimeMask:
    def __init__(self, mask_cfg=dict(), min_clear_ratio=0.0, max_clear_ratio=1.0):
        self.mask_generator = MaskGenerator(mask_cfg, min_clear_ratio, max_clear_ratio)

    def __call__(self, data_dict):
        ref_images = copy.deepcopy(data_dict["images"])
        num_frames = ref_images.shape[0]
        mask = self.mask_generator.get_mask(num_frames)
        ref_images = ref_images * (mask < 0.5)
        data_dict["ref_images"] = ref_images
        data_dict["time_mask"] = mask
        return data_dict


class GenerateFirstRefImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        return data_dict

class GenerateFirstAndLastRefImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        last_ref_image = copy.deepcopy(data_dict["images"][-1:, ...])
        data_dict["last_ref_image"] = last_ref_image
        return data_dict

class GenerateRepeatedFirstImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        return data_dict


class GeneratePoseControlImages:
    def __init__(self):
        pass

class GenerateRawFirstRefImage:
    def __call__(self, data_dict):
        raw_first_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["raw_first_image"] = raw_first_image
        return data_dict

class GenerateRawFirstLastRefImage:
    def __call__(self, data_dict):
        raw_first_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["raw_first_image"] = raw_first_image
        raw_last_image = copy.deepcopy(data_dict["images"][-1:, ...])
        data_dict["raw_last_image"] = raw_last_image
        return data_dict
    
@func_set_timeout(60)
class SampleImages:
    def __init__(
        self,
        num_frames=1,
    ):
        self.num_frames = num_frames

    @func_set_timeout(60)
    def __call__(self, data_dict):
        video = data_dict["video"]
        if self.num_frames > 1:
            sample_indexes = self.get_sample_indexes(data_dict, self.num_frames)
            images = video.get_frames_at(sample_indexes.tolist()).data
        else:
            images = np.array(video)
            images = torch.from_numpy(images).permute(2,0,1).unsqueeze(0).contiguous()
        data_dict["images"] = images
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        video_length = valid_range[1] - valid_range[0]

        frame_interval = data_dict["frame_interval"]
        sample_length = (num_frames - 1) * frame_interval + 1
        start_idx = valid_range[0] + random.randint(0, video_length - sample_length - 1)
        sample_indexes = np.linspace(
            start_idx, start_idx + sample_length - 1, num_frames, dtype=int
        )
        return sample_indexes
    
@func_set_timeout(60)
class SampleDynamicFPSVideo:
    def __init__(
        self,
        num_frames=1,
        max_frames=201,
        fps_config={"24": 1.0},
        default_fps=24,
    ):
        self.num_frames = num_frames
        self.fps_config = {}
        self.default_fps = default_fps
        self.max_frames = max_frames

        for k, v in fps_config.items():
            self.fps_config[int(k)] = v

        assert all(
            fps_ratio >= 0 for fps_ratio in self.fps_config.values()
        ), f"mask_ratio should be greater than or equal to 0, got {self.fps_config.values()}"
        assert all(
            fps_ratio <= 1 for fps_ratio in self.fps_config.values()
        ), f"mask_ratio should be less than or equal to 1, got {self.fps_config.values()}"
        # sum of mask_ratios should be 1
        assert math.isclose(
            sum(self.fps_config.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(self.fps_config.values())}"

    def __call__(self, data_dict):
        video = data_dict["video"]

        sample_indexes = self.get_sample_indexes(data_dict, self.num_frames)
        images = video.get_frames_at(sample_indexes.tolist()).data

        data_dict["images"] = images
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        
        fps_type = random.random()
        prob_acc = 0.
        dst_fps = self.default_fps
        for fps, ratio in self.fps_config.items():
            prob_acc = prob_acc + ratio
            if fps_type < prob_acc:
                dst_fps = fps
                break
        
        
        data_dict["dst_fps"] = dst_fps # inject fps
        data_dict["frame_interval"] = int(self.default_fps // dst_fps) # inject frame interval
        
        native_fps = data_dict['fps']
        extract_frame_interval = native_fps / dst_fps
        this_video_length = valid_range[1] - valid_range[0]
        
        data_dict["dst_fps"] = dst_fps # inject fps

        num_frames = int(1 + (this_video_length - 1) / extract_frame_interval)
        num_frames = max(1, (num_frames // 4 * 4) - 3)

        start_idx = valid_range[0]

        indexes = [start_idx + round(i * extract_frame_interval) for i in range(num_frames)]
        if len(indexes) > self.max_frames:
            start = random.randint(0, len(indexes) - self.max_frames)
            indexes = indexes[start: start + self.max_frames]
        sample_indexes = np.array(indexes, dtype=int)        

        return sample_indexes
    
@func_set_timeout(60)
class SampleWholeVideo:
    def __init__(
        self,
        max_frames=145,
        base_fps=24,
        fps_list=[24, 12, 6]
    ):
        self.max_frames = max_frames
        self.base_fps = base_fps
        self.fps_list = sorted(fps_list, reverse=True)
        
    def __call__(self, data_dict):
        video = data_dict["video"]

        sample_indexes, frame_interval = self.get_sample_indexes(data_dict)
        images = video.get_frames_at(sample_indexes.tolist()).data
        
        data_dict["images"] = images
        data_dict["frame_interval"] = frame_interval
        return data_dict

    def get_sample_indexes(self, data_dict):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        
        length = valid_range[1] - valid_range[0]
        
        native_fps = data_dict['fps']
        native_seconds = length / native_fps
        
        dst_fps = self.fps_list[-1]
        for fps in self.fps_list:
            if (self.max_frames / fps) > native_seconds:
                dst_fps = fps
                break

        if dst_fps > self.base_fps:
            dst_fps = self.base_fps
        
        frame_interval = native_fps / dst_fps
        
        data_dict["dst_fps"] = dst_fps # inject fps

        num_frames = int(1 + (length - 1) / frame_interval)
        num_frames = max(1, (num_frames // 4 * 4) - 3)

        start_idx = valid_range[0]

        indexes = [start_idx + round(i * frame_interval) for i in range(num_frames)]
        sample_indexes = np.array(indexes, dtype=int)        

        return sample_indexes, max(1, int(self.base_fps // dst_fps))
    
    
@func_set_timeout(60)
class SampleImageVideo:
    def __call__(self, data_dict):
        video = data_dict["video"]
        video_length = data_dict["video_info"][0]
        if video_length > 1:
            sample_indexes = self.get_sample_indexes(data_dict, video_length)
            images = video.get_frames_at(sample_indexes.tolist()).data
        else:
            images = np.array(video)
            images = torch.from_numpy(images).permute(2,0,1).unsqueeze(0).contiguous()
        data_dict["images"] = images
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        video_length = valid_range[1] - valid_range[0]

        frame_interval = data_dict["frame_interval"]
        sample_length = (num_frames - 1) * frame_interval + 1
        start_idx = valid_range[0] + random.randint(0, video_length - sample_length - 1)
        sample_indexes = np.linspace(
            start_idx, start_idx + sample_length - 1, num_frames, dtype=int
        )
        return sample_indexes
