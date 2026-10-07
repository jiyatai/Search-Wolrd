#!/usr/bin/env python
"""Export a stage-3 checkpoint into a deployment SearchWorldTrainer ckpt.

Stage 3 (value-guided imagination planning) trains ONLY the action head:
ImaginationRLModule.actor is a ContinuedActionPolicy holding deep copies of
the Stage-2 BC fusion + command MLP, refined by the value-guided objective
(pi+ reweighting) with stop held under the frozen pi_BC. Deployment
(evaluate_uav.py / SearchWorld.inference) calls world_model.action_policy, so
the continued head must be copied BACK into the world model's action_policy
before eval.

This script loads the stage-3 Lightning checkpoint, copies the trained
actor weights into a deployment SearchWorldTrainer checkpoint, and writes it out.

Mapping (ContinuedActionPolicy -> ActionPolicy inside world_model):
    actor.fusion.*      -> model.action_policy.policy_state_fusion.*
    actor.policy_mlp.*  -> model.action_policy.policy_mlp.*

Usage (inside container, CPU is fine):
    python scripts/export_stage3_policy.py \
        --stage3 /path/to/stage3/checkpoints/last.ckpt \
        --out /path/to/deploy_stage3.ckpt
"""
import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage3', required=True,
                        help='ImaginationRLModule checkpoint (.ckpt)')
    parser.add_argument('--out', required=True,
                        help='output deployment checkpoint (.ckpt)')
    args = parser.parse_args()

    print(f'loading {args.stage3} ...')
    ckpt = torch.load(args.stage3, map_location='cpu')
    sd = ckpt['state_dict']

    # Source: the continued actor weights (strip 'actor.' Lightning prefix).
    actor_keys = [k for k in sd if k.startswith('actor.')]
    if not any(k.startswith('actor.fusion.') for k in actor_keys):
        raise SystemExit(
            '[error] no actor.fusion.* keys found - the checkpoint was '
            'trained with use_fresh_actor=True (fresh ImagRLActor). The '
            'fresh head has a different architecture from the deployed '
            'ActionPolicy and cannot be exported by weight copy; retrain '
            'with use_fresh_actor=False, or switch the eval path to the '
            'ImagRLActor explicitly.')

    # Destination: the world-model ActionPolicy inside the same checkpoint
    # (the frozen stage-2 model rides along in every stage-3 ckpt).
    remap = {}
    n_fusion = n_mlp = 0
    for k in actor_keys:
        if k.startswith('actor.fusion.'):
            dst = 'world_model.action_policy.policy_state_fusion.' + \
                k[len('actor.fusion.'):]
            remap[dst] = k
            n_fusion += 1
        elif k.startswith('actor.policy_mlp.'):
            dst = 'world_model.action_policy.policy_mlp.' + \
                k[len('actor.policy_mlp.'):]
            remap[dst] = k
            n_mlp += 1
    print(f'[map] fusion params: {n_fusion}, mlp params: {n_mlp}')

    # Sanity: shapes must match exactly (they will - ContinuedActionPolicy
    # deep-copied the very modules it now replaces).
    for dst, src in remap.items():
        if dst not in sd:
            raise SystemExit(f'[error] missing dest key {dst}')
        if sd[dst].shape != sd[src].shape:
            raise SystemExit(
                f'[error] shape mismatch {dst} {tuple(sd[dst].shape)} '
                f'vs {src} {tuple(sd[src].shape)}')

    # Verify the continued head actually MOVED away from the frozen BC
    # original (otherwise stage 3 changed nothing and the export is a
    # no-op warning).
    moved = sum((sd[dst] - sd[src]).abs().max().item()
                for dst, src in remap.items())
    print(f'[check] max |continued - frozen_BC| summed over params: '
          f'{moved:.4e}')
    if moved == 0.0:
        print('[warn] continued head is IDENTICAL to the frozen BC head - '
              'stage 3 either did not train or the value guidance had no '
              'effect; export is a no-op.')

    # Copy continued weights over the world-model ActionPolicy.
    out_sd = dict(sd)
    for dst, src in remap.items():
        out_sd[dst] = sd[src].clone()

    # Keep only the deployment-relevant keys (drop actor. stage-3 head and
    # any legacy critic./pose_head./target_critic. keys; keep world_model.*
    # and Lightning bookkeeping) so SearchWorldTrainer.load_from_checkpoint
    # accepts it (strict=False in evaluate_uav.py tolerates the rest).
    drop_prefixes = ('actor.', 'critic.', 'pose_head.', 'target_critic.',
                     'engine.')
    out_sd = {k: v for k, v in out_sd.items()
              if not k.startswith(drop_prefixes)}

    slim = {
        'epoch': ckpt.get('epoch'),
        'global_step': ckpt.get('global_step'),
        'pytorch-lightning_version': ckpt.get('pytorch-lightning_version'),
        'state_dict': out_sd,
        'hyper_parameters': ckpt.get('hyper_parameters'),
        'hparams_name': ckpt.get('hparams_name'),
    }
    slim = {k: v for k, v in slim.items() if v is not None}
    tmp = args.out + '.tmp'
    torch.save(slim, tmp)
    import os
    os.replace(tmp, args.out)
    gb = os.path.getsize(args.out) / 1e9
    print(f'saved {args.out} ({gb:.2f} GB)')


if __name__ == '__main__':
    main()
