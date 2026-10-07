#!/usr/bin/env python
"""Remap stage-2 v2 checkpoint keys for the BEVDecoderMultiScale upgrade.

The old BEVDecoder used ``bev_decoder.decoder_layers.{i}.0`` (ConvTranspose2d)
and had no multi-scale heads. The new BEVDecoderMultiScale uses
``bev_decoder.decoder_blocks.{i}.0`` plus ``skip_convs.{i}`` and produces
multi-scale outputs. proj / final_conv keys are identical and carry over
unchanged; skip_convs are new and left to fresh initialisation.

Also strips 'losses.*' prefix keys (LightningModule hyper-parameter dumps)
that ``load_state_dict(strict=False)`` warns about.

Usage:
    python tools/remap_stage2_v2_ckpt.py \
        --ckpt /path/to/last.ckpt --output /path/to/last_ms.ckpt
"""
import argparse

import torch


CONV_REMAP = [
    ('bev_decoder.decoder_layers.', 'bev_decoder.decoder_blocks.'),
]


def remap_state_dict(state_dict: dict) -> dict:
    new_sd = {}
    n_renamed = 0
    n_dropped = 0
    for k, v in state_dict.items():
        # Drop LightningModule 'losses.*' hp-dump keys.
        if k.startswith('losses.'):
            n_dropped += 1
            continue
        new_k = k
        for old_prefix, new_prefix in CONV_REMAP:
            if new_k.startswith(old_prefix):
                new_k = new_prefix + new_k[len(old_prefix):]
                n_renamed += 1
                break
        new_sd[new_k] = v
    print(f'[remap] renamed {n_renamed} keys, dropped {n_dropped} keys, '
          f'{len(new_sd)} total')
    return new_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='source v2 last.ckpt')
    parser.add_argument('--output', required=True,
                        help='destination checkpoint path')
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    ckpt['state_dict'] = remap_state_dict(ckpt['state_dict'])
    torch.save(ckpt, args.output)
    print(f'[remap] saved -> {args.output}')


if __name__ == '__main__':
    main()
