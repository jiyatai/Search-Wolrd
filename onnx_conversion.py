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

import argparse
import os

import gin
import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from model.dataset.data_constants import INPUT_IMAGE_SIZE
from model.trainer import SearchWorldTrainer


class SearchWorldInference(pl.LightningModule):
    ''' Wrapper of SearchWorld for Onnx conversion.
    '''
    def __init__(self, checkpoint_path: str, enable_semantic: bool,
                 enable_rgb: bool, enable_depth: bool):
        super().__init__()
        self.searchworld = SearchWorldTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path)
        self.enable_semantic = enable_semantic
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        # Add policy noise as input for diffusion policy.
        if self.searchworld.model.action_policy.enable_policy_diffusion:
            self.register_buffer(
                "policy_noise",
                torch.randn((1,
                             self.searchworld.model.action_policy.
                             policy_diffuser.num_input_channels()),
                            dtype=torch.float32))

    def forward(self, image, route, speed, action_input, history_input,
                sample_input):
        inputs = {}
        # Resize the input image to desired size.
        image = image.squeeze(0)
        image = F.interpolate(image,
                              size=INPUT_IMAGE_SIZE,
                              mode='bilinear',
                              align_corners=False)
        inputs['image'] = image.unsqueeze(0)
        inputs['route'] = route
        inputs['speed'] = speed
        inputs['action'] = action_input
        inputs['history'] = history_input
        inputs['sample'] = sample_input
        if self.searchworld.model.action_policy.enable_policy_diffusion:
            inputs['policy_noise'] = self.policy_noise
        # Outputs: [action_output, history_output, sample_output, semantic_output, rgb_output]
        # "semantic_output" and "rgb_output" can be None depends on the input booleans.
        return self.searchworld.inference(inputs, self.enable_semantic,
                                         self.enable_rgb, self.enable_depth)


def convert(checkpoint_path: str, onnx_path: str, enable_semantic: bool,
            enable_rgb: bool, enable_depth: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SearchWorldInference(checkpoint_path, enable_semantic, enable_rgb,
                               enable_depth)

    model.to(device)
    model.eval()

    # Input tensors.
    # Note: Image size should be adjusted to match the camera resolution.
    image = torch.randn((1, 1, 3, 1200, 1920), dtype=torch.float32).to(device)
    speed = torch.randn((1, 1, 1), dtype=torch.float32).to(device)
    action = torch.randn((1, 6), dtype=torch.float32).to(device)
    history = torch.zeros((1, 1024), dtype=torch.float32).to(device)
    sample = torch.zeros((1, 512), dtype=torch.float32).to(device)
    route = torch.randn((1, 1, 19, 4), dtype=torch.float32).to(device)

    # Output names.
    output_names = [
        'action_output', 'path_output', 'history_output', 'sample_output'
    ]
    if enable_semantic:
        output_names.append('semantic_output')
    if enable_rgb:
        output_names.append('rgb_output')
    if enable_depth:
        output_names.append('depth_output')

    # ONNX conversion.
    torch.onnx.export(model, (image, route, speed, action, history, sample),
                      onnx_path,
                      verbose=True,
                      input_names=[
                          'image', 'route', 'speed', 'action_input',
                          'history_input', 'sample_input'
                      ],
                      output_names=output_names)


def main():
    # Parse the arguments.
    parser = argparse.ArgumentParser(
        description='Convert the SearchWorld to onnx.')
    parser.add_argument('--ckpt-path',
                        '-p',
                        type=str,
                        required=False,
                        help='The path to the checkpoint.')
    parser.add_argument('--ckpt-artifact',
                        '-a',
                        type=str,
                        required=False,
                        help='The wandb checkpoint artifact.')
    parser.add_argument('--onnx-file',
                        '-o',
                        type=str,
                        required=True,
                        help='The path to the output onnx file.')
    parser.add_argument('--enable-semantic',
                        action='store_true',
                        help="Enable semantic inference.")
    parser.add_argument('--enable-rgb',
                        action='store_true',
                        help='Enable rgb inference.')
    parser.add_argument('--enable-depth',
                        action='store_true',
                        help='Enable depth inference.')

    args = parser.parse_args()

    # Load hyperparameters.
    gin.parse_config_file('configs/train_config.gin', skip_unknown=True)

    # Sanity check on checkpoint input.
    if args.ckpt_path and args.ckpt_artifact:
        raise ValueError(
            'Both checkpoint path and checkpoint artifact are provided.')

    checkpoint_path = None
    if args.ckpt_path:
        checkpoint_path = args.ckpt_path
    elif args.ckpt_artifact:
        wandb_project = args.ckpt_artifact.split('/')[1]
        wandb_run_id = args.ckpt_artifact.split('/')[2].split('-')[1].split(
            ':')[0]
        run = wandb.init(dir='/tmp', project=wandb_project, id=wandb_run_id)
        checkpoint = run.use_artifact(args.ckpt_artifact, type='model')
        checkpoint_dir = checkpoint.download(root="/tmp")
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
    else:
        raise ValueError('No checkpoint provided.')

    # Run the convert.
    print("Converting ONNX.")
    convert(checkpoint_path, args.onnx_file, args.enable_semantic,
            args.enable_rgb, args.enable_depth)

    # Upload onnx to wandb if ckpt_artifact is provided.
    if args.ckpt_artifact:
        print('Uploading to WANDB.')
        version = args.ckpt_artifact.split('/')[2].split('-')[1].split(':')[1]
        onnx_artifact = wandb.Artifact(f'onnx-{wandb_run_id}-{version}',
                                       type='onnx')
        onnx_artifact.add_file(args.onnx_file)
        wandb.log_artifact(onnx_artifact)
        wandb.finish()


if __name__ == '__main__':
    main()
