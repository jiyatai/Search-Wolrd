# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

'''Stage-3 imagination RL training entry point.

Follows the same gin-loading order as train.py: ALL DataModules are
imported (registering their gin configurables) BEFORE gin.parse_config_file
is called, otherwise bindings like RSSM.* are silently skipped.

Usage (inside the container):
    python train_stage3.py \
        -c configs/stage3_imagination_config.gin \
        -d /workspace/datasets_shared/WorldSearch_data/expert_parquet \
        -o outputs/WorldSearch/stage3_imagination_rl \
        -n WorldSearch -r imagination_rl_v1
'''
import os
import sys
import signal

# Set NCCL timeout to be shorter (default is 30min, too long)
os.environ.setdefault('NCCL_TIMEOUT', '600')  # 10 minutes
os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')

# Set CUDA devices before importing torch
if '--gpus' in sys.argv:
    gpu_idx = sys.argv.index('--gpus')
    if gpu_idx + 1 < len(sys.argv):
        gpu_ids = sys.argv[gpu_idx + 1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"[train_stage3] Set CUDA_VISIBLE_DEVICES={gpu_ids}")
        sys.argv.pop(gpu_idx)
        sys.argv.pop(gpu_idx)

import gin
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from arg_parser import parse_arguments, TaskMode

# ---- gin loading order: import ALL DataModules FIRST (they register
# their gin configurables) BEFORE gin.parse_config_file is called.
from model.dataset.uav_dataset import UAVDataModule  # pylint: disable=unused-import
from model.dataset.uav_parquet_dataset import UAVParquetDataModule  # pylint: disable=unused-import
from model.trainer import SearchWorldTrainer  # pylint: disable=unused-import
from model.rl.trainer import ImaginationRLModule  # pylint: disable=unused-import


@gin.configurable
def train(dataset_path, output_dir, ckpt_path, wandb_entity_name,
          wandb_project_name, wandb_run_name, precision, epochs,
          data_module, devices=None, limit_train_batches=None,
          limit_val_batches=None,
          # Accepted (and ignored) for compatibility with the shared
          # base_train_config.gin include, which binds these for train.py.
          model_trainer=None, ckpt_skip_n_epochs=None):
    # Auto-detect devices from env or use all GPUs
    if devices is None or devices == 'auto':
        _cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if _cvd:
            devices = len([x for x in _cvd.split(',') if x.strip()])
        else:
            import torch
            devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"[train_stage3] Using devices={devices}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = data_module(dataset_path=dataset_path)
    model = ImaginationRLModule()

    wandb_logger = WandbLogger(entity=wandb_entity_name,
                               project=wandb_project_name,
                               name=wandb_run_name,
                               save_dir=output_dir,
                               group='DDP',
                               log_model=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        save_top_k=2,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )

    callbacks = [
        pl.callbacks.ModelSummary(-1),
        pl.callbacks.LearningRateMonitor(),
        checkpoint_callback,
    ]

    def cleanup_handler(signum, frame):
        print(f"\n[train_stage3] Received signal {signum}, cleaning up...")
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        sys.exit(1)

    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)

    trainer = pl.Trainer(max_epochs=epochs,
                         precision=precision,
                         limit_train_batches=limit_train_batches,
                         limit_val_batches=limit_val_batches,
                         # Stabiliser #2: global-norm gradient clipping.
                         # Run v1 diverged with fp16 gradients that reached
                         # ~1e4+ norm before the value explosion; clip=1.0
                         # bounds every update regardless of loss scale.
                         gradient_clip_val=1.0,
                         sync_batchnorm=True,
                         callbacks=callbacks,
                         # The frozen world model's forward graph has unused
                         # parameters (e.g. semantic heads) under no_grad /
                         # eval usage patterns.
                         strategy='ddp_find_unused_parameters_true',
                         devices=devices,
                         accelerator='gpu',
                         logger=wandb_logger)
    trainer.fit(model, datamodule=data)

    return wandb_logger


def main():
    args = parse_arguments(TaskMode.TRAIN)

    for config_file in args.config_files:
        gin.parse_config_file(config_file, skip_unknown=True)

    wandb_logger = train(args.dataset_path, args.output_dir,
                         args.checkpoint_path, args.wandb_entity_name,
                         args.wandb_project_name, args.wandb_run_name)

    # Log the operative gin config for reproducibility.
    gin_config_str = gin.operative_config_str()
    with open('/tmp/gin_config_stage3.txt', 'w', encoding='UTF-8') as f:
        f.write(gin_config_str)
    artifact = wandb.Artifact('gin_config_stage3', type='text')
    artifact.add_file('/tmp/gin_config_stage3.txt')
    wandb_logger.experiment.log_artifact(artifact)

    wandb.finish()


if __name__ == '__main__':
    main()
