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
import logging
import time
import os
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import deepspeed

import torch
from torch import distributed as dist
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler, DistEpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.datasets.lerobot_dataset import MultiDatasetforDistTraining
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from lerobot.common.datasets.transforms import ImageTransforms

def init_logger(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if int(os.environ["RANK"]) == 0:
        formatter = logging.Formatter(
            f'[%(asctime)s] [rank: {int(os.environ["RANK"])}] [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件Handler
        log_path = Path(cfg.log_dir) / f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
        
def load_training_state(checkpoint_path, optimizer, lr_scheduler, accelerator):
    # 加载accelerate状态
    accelerator.load_state(checkpoint_path)
    
    # 加载额外元数据
    metadata = torch.load(checkpoint_path / "metadata.pt")
    step = metadata["step"]
    
    # 恢复优化器和学习率调度器状态
    if optimizer is not None:
        optimizer.load_state_dict(accelerator.get_optimizer_state(optimizer))
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(accelerator.get_scheduler_state(lr_scheduler))
    
    return step, optimizer, lr_scheduler


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
    )
    
    print(image_transforms, cfg.dataset.image_transforms)
    
    deepspeed.init_distributed()
    
    logger = init_logger(cfg)
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + int(os.environ.get('RANK', 0)))
    
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(pformat(cfg.to_dict()))
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(cfg)
        else:
            wandb_logger = None
            logger.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    else:
        wandb_logger = None

    if cfg.seed is not None:
        set_seed(cfg.seed + int(os.environ.get('RANK', 0)))  # Add process index for deterministic seeding

    # Dataset and policy setup
    # dataset = MultiDatasetforDistTraining(cfg=cfg, image_transforms=image_transforms, 
    #                        seed=cfg.seed, data_mix="oxe_magic_soup_plus",
    #                        vla2root_json="vla2root.json")
    dataset = MultiDatasetforDistTraining(cfg=cfg, image_transforms=image_transforms, 
                           seed=cfg.seed, data_mix="oxe_magic_soup_plus",
                           vla2root_json="vla2root_bak_single.json")
    logger.info(f"Dataset Meta: {dataset.meta}")

    # Policy setup
    logger.info("Creating policy...")
    if hasattr(cfg.policy, "tokenizer_max_length"):
        logger.info("Setiing model's tokenizer_max_length to 65")
        cfg.policy.tokenizer_max_length=65
    logger.info("Still creating policy...")
    policy = make_policy(
        cfg=cfg.policy,
        device=torch.cuda.current_device(),
        ds_meta=dataset.meta,
    )
    logger.info("Policy model created...")

    # Environment setup (only in main process)
    eval_env = None
    if int(os.environ.get('RANK', 0))==0 and cfg.eval_freq > 0 and cfg.env is not None:
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    # Optimizer and scheduler
    logger.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Resume training state
    step = 0
    # if cfg.resume:
    #     accelerator.load_state(cfg.checkpoint_path)
    #     if accelerator.is_main_process:
    #         metadata = torch.load(cfg.checkpoint_path / "metadata.pt")
    #         step = int(metadata["step"])
    #         # 广播step到所有进程
    #         step_tensor = torch.tensor([step], device=accelerator.device)
    #         torch.distributed.broadcast(step_tensor, src=0)
    #         step = step_tensor.item()
    #     else:
    #         # 非主进程接收step值
    #         step_tensor = torch.tensor([0], device=accelerator.device)
    #         torch.distributed.broadcast(step_tensor, src=0)
    #         step = step_tensor.item()
        

    # Logging setup (main process only)
    if int(os.environ.get('RANK', 0))==0:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logger.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logger.info(f"{cfg.env.task=}")
        logger.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logger.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logger.info(f"{dataset.num_episodes=}")
        logger.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logger.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Dataloader setup
    if hasattr(cfg.policy, "drop_n_last_frames"):
        logger.info("Creating EpisodeAwareSampler with drop_n_last_frames")
        sampler = DistEpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
            num_replicas=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"])
        )
    else:
        logger.info("Creating DistributedSampler")
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
            shuffle=True,
            seed=cfg.seed
        )
        shuffle = False

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=cfg.batch_size,
    #     sampler=sampler,
    #     num_workers=cfg.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    # Prepare components with Accelerator
    model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
        model=policy,
        optimizer=optimizer,
        training_data=dataset,
        lr_scheduler=lr_scheduler,
        config=cfg.deepspeed,
        model_parameters=policy.parameters(),
    )
    dl_iter = cycle(dataloader)
    
    total_steps = cfg.steps * cfg.gradient_accumulation_steps
    completed_steps = step * cfg.gradient_accumulation_steps

    # Metrics setup
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "optim_s": AverageMeter("optim_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size * int(os.environ["WORLD_SIZE"]) * cfg.gradient_accumulation_steps,  # Total batch size across all processes
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step
    )

    # Main training loop
    dist.barrier()
    # world_size = int(os.environ["WORLD_SIZE"])
    # logger.info(f"Start training on {world_size} devices")
    total_steps = cfg.steps * cfg.gradient_accumulation_steps
    completed_steps = step * cfg.gradient_accumulation_steps
    for _ in range(completed_steps, total_steps):
        dataloading_s = 0
        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_time = time.perf_counter() - start_time
        dataloading_s += dataloading_time

        # rank = os.environ.get('RANK')
        # local_rank = os.environ.get('LOCAL_RANK')
        # node_rank = os.environ.get('NODE_RANK')
        # world_size = os.environ.get('WORLD_SIZE')
        # maddr = os.environ.get('MASTER_ADDR')
        # mport = os.environ.get('MASTER_PORT')
        
        # print(f"rank: {rank}, local_rank: {local_rank}, node_rank: {node_rank}, world_size: {world_size}, maddr: {maddr}, mport: {mport}")
        
        for key in batch:
            
            if isinstance(batch[key], torch.Tensor):
                print(key, batch[key].shape)
            elif isinstance(batch[key], list):
                print(key, len(batch[key]))
                print(f"example {key} 0:", batch[key][0])
            else:
                print(key, type(batch[key]))
        # batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # for i in tqdm(range(100)):
        #     loss, output_dict = model_engine(batch)
        # print("\nlosses:", loss)
        # print("\noutput_dict:", output_dict)
        break
        
    # Destroy process group
    # deepspeed.destroy_process_group()
    dist.destroy_process_group()

if __name__ == "__main__":
    train()