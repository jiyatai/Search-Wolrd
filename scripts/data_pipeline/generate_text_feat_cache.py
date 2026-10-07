"""Precompute SigLIP text features for the expert parquet dataset.

Scans all *_metadata.json files under <output>/{train,val,test}, encodes each
unique task_description with the frozen SigLIP2 text encoder, and saves
<output>/text_feat_cache_{split}.pt files compatible with
UAVParquetDataset's cache lookup (root-level text_feat_cache_<split>.pt).

Usage:
    python scripts/generate_text_feat_cache.py \
        --dataset /path/to/expert_parquet \
        [--model google/siglip2-base-patch16-224]
"""
import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def extract_tensor(out):
    """Handle both old (<5, returns tensor) and new (>=5, ModelOutput) transformers."""
    if torch.is_tensor(out):
        return out
    if getattr(out, 'pooler_output', None) is not None:
        return out.pooler_output
    if getattr(out, 'text_embeds', None) is not None:
        return out.text_embeds
    raise ValueError(f"Cannot extract text embedding from {type(out)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True,
                        help='Parquet dataset root (contains train/val/test)')
    parser.add_argument('--model', default='google/siglip2-base-patch16-224')
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    # Collect unique descriptions per split
    split_descs = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(args.dataset, split)
        if not os.path.isdir(split_dir):
            continue
        descs = set()
        for meta_path in glob.glob(os.path.join(split_dir, 'ep_*',
                                                '*_metadata.json')):
            with open(meta_path) as f:
                d = json.load(f)
            desc = d.get('task_description')
            if desc:
                descs.add(desc)
        split_descs[split] = sorted(descs)
        print(f'{split}: {len(descs)} unique descriptions')

    all_descs = sorted(set().union(*split_descs.values())) if split_descs else []
    print(f'total unique: {len(all_descs)}')

    # Encode all unique descriptions
    feats = {}
    with torch.no_grad():
        for desc in all_descs:
            t = tok(text=desc, return_tensors='pt', padding=True,
                    truncation=True, max_length=64)
            out = model.get_text_features(**t)
            feats[desc] = F.normalize(extract_tensor(out), dim=-1).cpu()
    dim = next(iter(feats.values())).shape[-1]
    print(f'encoded dim: {dim}')

    # Save per-split caches
    for split, descs in split_descs.items():
        if not descs:
            continue
        cache = {
            'text_feat_dim': dim,
            'text_feat': {d: feats[d] for d in descs},
        }
        out_path = os.path.join(args.dataset, f'text_feat_cache_{split}.pt')
        torch.save(cache, out_path)
        print(f'saved: {out_path} ({len(descs)} entries)')


if __name__ == '__main__':
    main()
