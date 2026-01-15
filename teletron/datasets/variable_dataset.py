from typing import List
import numpy as np
from .utils import image_utils
from .base_dataset import BaseDataset
from teleai_data_tool.schema.dataset import ClipsDataset as _ClipsDataset
from teleai_data_tool.schema.clip import Clip
from teleai_data_tool.file.lmdb_client import LmdbClient
from teleai_data_tool.file.file_client import FileClient
import json
from teleai_data_tool.logger import logger
from collections import defaultdict
from tqdm import tqdm
from cattrs import structure
import random


class VariableClipDataset(BaseDataset):
    def __init__(
        self,
        data_path_list,
        transforms,
        filter_cfg=dict(),
        data_weight_list=[],
        enable_bucket_index=True,
        serialize_data=True,
    ) -> None:
        self.data_path_list = data_path_list
        self.data_weight_list = data_weight_list
        self.enable_bucket_index = enable_bucket_index
        self.bucket_index_list = None
        super().__init__(
            ann_file="",
            serialize_data=serialize_data,
            test_mode=False,
            lazy_init=False,
            max_refetch=10,
            pipeline=transforms,
            filter_cfg=filter_cfg,
        )
        self.file_client = FileClient()
        self.lmdb_client = LmdbClient()

    def load_data_list(self) -> List[dict]:
        data_list = []
        for data_path in tqdm(self.data_path_list):
            with open(data_path) as f:
                dataset = json.load(f)
            for clip in dataset["clips"]:
                clip = structure(clip, Clip)
                clip.file_path = f"{dataset['clip_data_root']}:{clip.file_path}"
                clip.meta["data_format"] = dataset["clip_data_type"]
                data_list.append(clip)
        return data_list

    def get_bucket_index_list(self):
        return self.bucket_index_list

    def _rand_another(self, bucket_idx) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.choice(self.bucket_index_list[bucket_idx])
    
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
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            logger.info(
                "Please call `full_init()` method manually to accelerate " "the speed."
            )
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception(
                    "Test time pipline should not get `None` " "data_sample"
                )
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                bucket_idx = self.data_list[idx].bucket_index
                idx = self._rand_another(bucket_idx)
                continue
            return data

        raise Exception(
            f"Cannot find valid image after {self.max_refetch}! "
            "Please check your image path and pipeline"
        )
    
    def filter_data(self):
        dst_size = self.filter_cfg.get("dst_size", (720, 480))
        dst_num_frames = self.filter_cfg.get("dst_num_frames", 100)
        dst_fps = self.filter_cfg.get("dst_fps", 24)
        multiple = self.filter_cfg.get("multiple", 16)
        min_area = self.filter_cfg.get("min_area", dst_size[0] * dst_size[1])
        optical_flow_th = self.filter_cfg.get("optical_flow_th", 2)
        aesthetic_th = self.filter_cfg.get("aesthetic_th", 4)
        bucket_size_th = self.filter_cfg.get("bucket_size_th", 4)
        motion_th = self.filter_cfg.get("motion_th", 0) 
        clearity_th = self.filter_cfg.get("clearity_th", 0.8) 
        laplacian_th = self.filter_cfg.get("laplacian_th", 0)
        training_suitability_th = self.filter_cfg.get("training_suitability_th", 3.7) 
        area_th = self.filter_cfg.get("area_th", 0)
        # fileter tag 
        too_small = 0
        too_short = 0
        motion_mismatch = 0
        aes_mismatch = 0
        motion_mismatch = 0
        clearity_mismatch = 0
        motion_mismatch = 0
        suitability_mismatch = 0
        buckets_mismatch = 0
        new_data_list = []
        shape_list = []
        shape_num_map = defaultdict(int)
        broken_clip = 0

        # prepare bucket
        self.buckets_size = self.filter_cfg.get("buckets_size", [(640, 368), (864, 480), (960, 528)])
        self.buckets_size_ratio = self.filter_cfg.get("buckets_size_ratio", [0.3, 0.4, 0.3])
        for shape in self.buckets_size:
            shape_list.append(f"{shape[0]}__{shape[1]}")

        for clip in self.data_list:
            frame_interval = max(1, round(clip.fps / dst_fps))
            min_num_frames = frame_interval * dst_num_frames
            setattr(clip, "frame_interval", frame_interval)
            setattr(clip, "min_num_frames", min_num_frames)
            
            if clip.caption is None:
                broken_clip += 1
                continue
            # length
            if clip.length < min_num_frames:
                too_short += 1
                continue
            # size
            if clip.height * clip.width < min_area:
                too_small += 1
                continue

            if (clip.filter_state is not None):
                # aesthetic
                if (
                    clip.filter_state.aesthetic is None
                    or clip.filter_state.aesthetic < aesthetic_th
                ):
                    aes_mismatch += 1
                    continue

                # laplacian, 部分数据没有laplacian，所以这里是 and
                if (
                    clip.filter_state.laplacian is not None
                    and clip.filter_state.laplacian < laplacian_th
                ):
                    clearity_mismatch += 1
                    continue

                # optical_flow
                if clip.filter_state.optical_flow != -1.0:
                    if (
                        clip.filter_state.optical_flow is None
                        or clip.filter_state.optical_flow < optical_flow_th
                    ):
                        motion_mismatch += 1
                        continue
            
                # size
                if clip.filter_state.area < area_th:
                    too_small += 1
                    continue

                # clearity
                if (
                    clip.filter_state.clearity is not None
                    and clip.filter_state.clearity < clearity_th
                ):
                    clearity_mismatch += 1
                    continue

                # motion
                if (
                    clip.filter_state.motion is not None
                    and clip.filter_state.motion < motion_th
                ):
                    motion_mismatch += 1
                    continue

                # training_suitability
                if (
                    clip.filter_state.video_training_suitability is not None
                    and clip.filter_state.video_training_suitability < training_suitability_th
                ):
                    suitability_mismatch += 1
                    continue

            if self.enable_bucket_index:
                sampler_size = random.choices(self.buckets_size, weights=self.buckets_size_ratio)[0]
                dst_width, dst_height = image_utils.get_image_size(
                    (clip.width, clip.height),
                    (sampler_size[0], sampler_size[1]),
                    mode="area",
                    multiple=multiple,
                )
                video_info = f"{sampler_size[0]}__{sampler_size[1]}"
                setattr(clip, "bucket_index", shape_list.index(video_info))
                shape_num_map[video_info] += 1
                setattr(clip, "video_info", (dst_width, dst_height))
            new_data_list.append(clip)
        if self.enable_bucket_index:
            invalid_bucket_id_list = []
            for k, v in shape_num_map.items():
                if v < bucket_size_th:
                    buckets_mismatch += v
                    invalid_bucket_id_list.append(k)
            valid_data_list = [
                clip
                for clip in new_data_list
                if shape_list[clip.bucket_index] not in invalid_bucket_id_list
            ]
            bucket_index_list = defaultdict(list)
            for i, clip in enumerate(valid_data_list):
                bucket_index_list[clip.bucket_index].append(i)
            self.bucket_index_list = [
                np.array(bucket_index_list[i]) for i in range(len(shape_list))
            ]
        else:
            valid_data_list = new_data_list
        print(
            f"finish filter dataset, from {len(self.data_list)} to {len(valid_data_list)} \n"
            f"too short data {too_short} \n"
            f"too small data {too_small} \n"
            f"motion mismatch data {motion_mismatch} \n"
            f"aesthetic mismatch data {aes_mismatch} \n"
            f"clearity score mismatch data {clearity_mismatch} \n"
            f"suitability score mismatch data {suitability_mismatch} \n"
            f"buckets mismatch data {buckets_mismatch} \n"
            f"broken clip: {broken_clip} \n" 
            f"bucket shape: {shape_num_map}" 
        )
        return valid_data_list

    def get_data_info(self, idx):
        clip: Clip = super().get_data_info(idx)
        data_dict = dict()
        if clip.meta["data_format"] == "lmdb":
            video = self.lmdb_client.get(clip.file_path, num_threads=8)
        elif clip.meta["data_format"] == "file":
            video = self.file_client.get(clip.file_path, num_threads=8)
        data_dict["clip_info"] = clip
        data_dict["video"] = video
        data_dict["video_info"] = clip.video_info
        data_dict["video_length"] = clip.length
        data_dict["video_height"] = clip.height
        data_dict["video_width"] = clip.width
        data_dict["slice_index"] = None
        if len(clip.caption.frame_range) > 0:
            last_slice = clip.caption.frame_range[-1]
            slice_length = len(clip.caption.frame_range)
            if (last_slice[1] - last_slice[0]) < clip.min_num_frames:
                slice_length = slice_length-1
            slice_index = random.randint(0, slice_length-1)
            data_dict["video_valid_range"] = clip.caption.frame_range[slice_index]
            data_dict["slice_index"] = slice_index
        else:
            data_dict["video_valid_range"] = clip.valid_range
        data_dict["fps"] = clip.fps
        data_dict["frame_interval"] = clip.frame_interval
        data_dict["bucket_index"] = clip.bucket_index
        return data_dict
