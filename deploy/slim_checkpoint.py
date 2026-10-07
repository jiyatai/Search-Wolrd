#!/usr/bin/env python
"""Strip optimizer/scheduler states from a training checkpoint.

4.29GB full ckpt  ->  ~2.5GB inference-only ckpt.
Keeps: state_dict, hyper_parameters (needed by load_from_checkpoint).

Usage:
    python slim_checkpoint.py <in.ckpt> <out.ckpt>
"""
import sys
import os
import torch

KEEP_KEYS = (
    'epoch',
    'global_step',
    'pytorch-lightning_version',
    'state_dict',
    'hyper_parameters',
    'hparams_name',
    'MixedPrecision',
)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]

    print(f'loading {src} ...')
    ckpt = torch.load(src, map_location='cpu')

    slim = {k: ckpt[k] for k in KEEP_KEYS if k in ckpt}
    n_params = sum(v.numel() for v in slim['state_dict'].values())
    print(f'params: {n_params / 1e6:.1f}M')

    tmp = dst + '.tmp'
    torch.save(slim, tmp)
    os.replace(tmp, dst)

    gb = os.path.getsize(dst) / 1e9
    print(f'saved {dst} ({gb:.2f}GB)')


if __name__ == '__main__':
    main()
