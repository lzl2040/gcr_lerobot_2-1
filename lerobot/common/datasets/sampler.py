#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterator, Union

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class DistEpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        # 新增分布式参数
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ):
        """Sampler that incorporates episode boundary information with distributed support.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            num_replicas: Number of processes participating in distributed training.
            rank: Rank of the current process.
            seed: Random seed for reproducibility.
        """
        # 原始索引生成逻辑保持不变
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        # 分布式分片逻辑
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0  # 用于shuffle同步
        self.seed = seed
        
        # 计算分片索引范围
        total_size = len(indices)
        if total_size == 0:
            raise ValueError("No indices available after filtering")
            
        # 每卡样本数计算（处理余数）
        num_samples = total_size // num_replicas
        remainder = total_size % num_replicas
        self.num_samples = num_samples + (1 if rank < remainder else 0)
        self.total_size = self.num_samples * num_replicas
        
        # 确定当前卡的索引范围
        start_idx = rank * num_samples + min(rank, remainder)
        end_idx = start_idx + self.num_samples
        self.indices = indices[start_idx:end_idx]
        
        # 保持shuffle功能
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        # 带同步的随机shuffle
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)  # 保持各卡shuffle同步
            shuffled = torch.randperm(len(self.indices), generator=g).tolist()
            indices = [self.indices[i] for i in shuffled]
        else:
            indices = self.indices
            
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        """用于同步不同进程的shuffle"""
        self.epoch = epoch