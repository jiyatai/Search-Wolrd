# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import signal

# Set NCCL timeout to be shorter (default is 30min, too long)
os.environ.setdefault('NCCL_TIMEOUT', '600')  # 10 minutes
os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')

# Set CUDA devices before importing torch
# Priority: 1) --gpus flag  2) CUDA_VISIBLE_DEVICES env  3) all GPUs
if '--gpus' in sys.argv:
    gpu_idx = sys.argv.index('--gpus')
    if gpu_idx + 1 < len(sys.argv):
        gpu_ids = sys.argv[gpu_idx + 1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"[train.py] Set CUDA_VISIBLE_DEVICES={gpu_ids}")
        sys.argv.pop(gpu_idx)
        sys.argv.pop(gpu_idx)

# Detect actual CUDA_VISIBLE_DEVICES
_cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if _cvd:
    _gpu_count = len([x for x in _cvd.split(',') if x.strip()])
    print(f"[train.py] CUDA_VISIBLE_DEVICES={_cvd} -> PyTorch will see {_gpu_count} GPU(s)")
else:
    _gpu_count = None
    print(f"[train.py] No CUDA_VISIBLE_DEVICES set, will use all GPUs")

import gin
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from arg_parser import parse_arguments, TaskMode

from model.dataset.uav_dataset import UAVDataModule  # pylint: disable=unused-import
from model.dataset.uav_parquet_dataset import UAVParquetDataModule  # pylint: disable=unused-import
from model.trainer import SearchWorldTrainer  # pylint: disable=unused-import


@gin.configurable
def train(dataset_path, output_dir, ckpt_path, wandb_entity_name,
          wandb_project_name, wandb_run_name, precision, epochs, data_module,
          model_trainer, devices=None, ckpt_skip_n_epochs=30,
          limit_train_batches=None, limit_val_batches=None,
          accumulate_grad_batches=1):
    # Auto-detect devices from env or use all GPUs
    if devices is None or devices == "auto":
        _cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if _cvd:
            devices = len([x for x in _cvd.split(',') if x.strip()])
        else:
            import torch
            devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"[train.py] Using devices={devices}")

    # Create a output directory if not exit.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = data_module(dataset_path=dataset_path)
    if ckpt_path:
        model = model_trainer.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                   strict=False)
    else:
        model = model_trainer()

    wandb_logger = WandbLogger(entity=wandb_entity_name,
                               project=wandb_project_name,
                               name=wandb_run_name,
                               save_dir=output_dir,
                               group="DDP",
                               log_model=False)  # Disable model upload

    # Custom checkpoint: skip first N epochs, then save top 2 + last
    class CustomModelCheckpoint(ModelCheckpoint):
        def __init__(self, skip_n_epochs=30, **kwargs):
            super().__init__(**kwargs)
            self.skip_n_epochs = skip_n_epochs

        def _should_save_on_val_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch < self.skip_n_epochs:
                return False
            return super()._should_save_on_val_epoch_end(trainer, pl_module)

    checkpoint_callback = CustomModelCheckpoint(
        skip_n_epochs=ckpt_skip_n_epochs,
        dirpath=os.path.join(output_dir, 'checkpoints'),
        save_top_k=2,
        monitor='val_loss',
        mode='min',
        save_last=True,
    )
    print(f"[train.py] checkpoint: skip_n_epochs={ckpt_skip_n_epochs}, "
          f"limit_train_batches={limit_train_batches}, "
          f"limit_val_batches={limit_val_batches}")

    callbacks = [
        pl.callbacks.ModelSummary(-1),
        pl.callbacks.LearningRateMonitor(),
        checkpoint_callback,
    ]

    # Add signal handler to clean up on interrupt
    def cleanup_handler(signum, frame):
        print(f"\n[train.py] Received signal {signum}, cleaning up...")
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
                         accumulate_grad_batches=accumulate_grad_batches,
                         sync_batchnorm=True,
                         callbacks=callbacks,
                         strategy='ddp_find_unused_parameters_true',
                         devices=devices,
                         accelerator="gpu",
                         logger=wandb_logger)
    trainer.fit(model, datamodule=data)

    trainer.test(ckpt_path="last", datamodule=data)

    return wandb_logger


def log_gin_config(logger: WandbLogger):
    # This function should be called after all the gin configurable functions.
    # Otherwise, the config string will be empty.
    gin_config_str = gin.operative_config_str()

    # Create a temporary file to store the gin config
    with open("/tmp/gin_config.txt", "w", encoding='UTF-8') as f:
        f.write(gin_config_str)

    # Log the artifact using the WandbLogger
    artifact = wandb.Artifact("gin_config", type="text")
    artifact.add_file("/tmp/gin_config.txt")
    logger.experiment.log_artifact(artifact)


def main():
    args = parse_arguments(TaskMode.TRAIN)

    for config_file in args.config_files:
        gin.parse_config_file(config_file, skip_unknown=True)

    # Run the training loop.
    wandb_logger = train(args.dataset_path, args.output_dir,
                         args.checkpoint_path, args.wandb_entity_name,
                         args.wandb_project_name, args.wandb_run_name)

    # Log gin config
    log_gin_config(wandb_logger)

    # Finish wandb
    wandb.finish()


if __name__ == '__main__':
    main()
