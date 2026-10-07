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
from enum import Enum


class TaskMode(Enum):
    TRAIN = "train"
    EVAL = "eval"


def parse_arguments(task_mode):
    ''' Arguments parser for model training and evaluation.
    '''
    description = 'Train SearchWorld' if task_mode == TaskMode.TRAIN else 'Eval SearchWorld'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--config-files',
                        '-c',
                        nargs='+',
                        required=True,
                        help='The list of the config files.')
    parser.add_argument('--dataset-path',
                        '-d',
                        type=str,
                        required=True,
                        help='The path to the dataset.')
    parser.add_argument('--wandb-entity-name',
                        '-e',
                        type=str,
                        help='The entity name of W&B.')
    parser.add_argument('--wandb-project-name',
                        '-n',
                        type=str,
                        default='searchworld_train'
                        if task_mode == TaskMode.TRAIN else 'searchworld_eval',
                        help='The project name of W&B.')
    parser.add_argument(
        '--wandb-run-name',
        '-r',
        type=str,
        default='train_run' if task_mode == TaskMode.TRAIN else 'eval_run',
        help='The run name of W&B.')
    parser.add_argument('--checkpoint-path',
                        '-p',
                        type=str,
                        default=None,
                        help='The path to the checkpoint.')
    if task_mode == TaskMode.TRAIN:
        parser.add_argument('--output-dir',
                            '-o',
                            type=str,
                            required=True,
                            help='The path to the output dir.')

    # NOTE: TaskMode.EVAL is reserved for the closed-loop UAV-ON entry point
    # (see deploy/README.md §5).  Offline evaluation uses evaluate_uav.py,
    # which takes its arguments directly and therefore needs no eval flags
    # here.

    args = parser.parse_args()
    return args
