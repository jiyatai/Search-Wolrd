#!/usr/bin/env python
"""Warm-start remap: v2 BEVDecoder -> v3 BEVDecoderMultiScale.

The v2 checkpoint (e.g. stage2_expert_v2/last.ckpt, epoch 49) stores the old
single-scale BEVDecoder trunk as a FLAT nn.Sequential:
    model.bev_decoder.proj.0.*                        (Linear 1536 -> 512*8*8)
    model.bev_decoder.decoder_layers.{0,3,6,9}.*      (4x ConvTranspose2d)
    model.bev_decoder.decoder_layers.{1,4,7,10}.*     (4x BatchNorm2d)
    model.bev_decoder.final_conv.*                    (ConvT -> 3ch)

The v3 model (BEVDecoderMultiScale) keeps `proj` / `final_conv` names but
restructures the trunk into `decoder_blocks.{i}.0` (ConvT) /
`decoder_blocks.{i}.1` (BatchNorm) and adds 4 new `skip_convs.{i}` (1x1 conv)
that stay randomly initialised. `strict=False` load will NOT remap trunk
weights automatically, so without this script the whole trunk restarts from
scratch.

Usage (inside the training container, cwd=/workspace/SearchWorld):
    python scripts/remap_v2_to_v3_bev.py \
        --src ../datasets_shared/WorldSearch_data/experiments/stage2_expert_v2/checkpoints/last.ckpt \
        --dst ../datasets_shared/WorldSearch_data/experiments/stage3_smoke/init_from_v2_ms.ckpt
"""
import argparse
import os

import torch

# Flat v2 Sequential index -> (v3 block index, position within block)
# v2: decoder_layers.0 (ConvT 512->256), 1 (BN), 3 (ConvT 256->128), 4 (BN),
#     6 (ConvT 128->64), 7 (BN), 9 (ConvT 64->32), 10 (BN)
# v3: decoder_blocks.{i}.0 (ConvT), decoder_blocks.{i}.1 (BN)
V2_TO_V3 = {
    '0': ('0', '0'), '1': ('0', '1'),
    '3': ('1', '0'), '4': ('1', '1'),
    '6': ('2', '0'), '7': ('2', '1'),
    '9': ('3', '0'), '10': ('3', '1'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='v2 checkpoint (last.ckpt)')
    parser.add_argument('--dst', required=True, help='output warm-start checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                        help='only print which keys will be remapped')
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location='cpu')
    sd = ckpt['state_dict']

    new_sd = {}
    remapped = []
    for k, v in sd.items():
        # Checkpoint keys carry the 'model.' prefix (SearchWorldTrainer.model).
        # 'model.bev_decoder.decoder_layers.{idx}.{param}'
        #     -> 'model.bev_decoder.decoder_blocks.{block}.{pos}.{param}'
        if 'decoder_layers.' in k:
            head, tail = k.split('decoder_layers.', 1)
            idx, param = tail.split('.', 1)
            if idx in V2_TO_V3:
                block, pos = V2_TO_V3[idx]
                new_k = f'{head}decoder_blocks.{block}.{pos}.{param}'
            else:
                new_k = k  # ReLU etc. have no params; keep as-is
            new_sd[new_k] = v
            remapped.append((k, new_k))
        else:
            new_sd[k] = v

    print(f'total keys: {len(sd)}')
    print(f'remapped decoder_layers -> decoder_blocks: {len(remapped)} keys')
    for old, new in remapped:
        print(f'  {old}  ->  {new}')

    # Sanity check: names the v3 model expects.
    expect = ['model.bev_decoder.proj.0.weight',
              'model.bev_decoder.final_conv.weight',
              'model.bev_decoder.decoder_blocks.0.0.weight',
              'model.bev_decoder.decoder_blocks.3.0.weight']
    for e in expect:
        print(f'  [{"OK" if e in new_sd else "MISSING"}] {e}')

    if args.dry_run:
        print('dry-run: not writing.')
        return

    ckpt['state_dict'] = new_sd
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f'saved: {args.dst}')


if __name__ == '__main__':
    main()
