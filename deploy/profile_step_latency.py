#!/usr/bin/env python3
"""Profile per-step decision latency of the SearchWorld agent.

Purpose: fill Table 4 (tab:efficiency) with measured per-step / per-episode
latency. Times the deployed inference path of SearchWorld.inference():
    UAVObservationEncoder (frozen DINOv2-S + SigLIP2 + pose MLP + fusion)
    -> RSSM.observe_step -> ActionPolicy (MLP) -> BEVDecoderMultiScale
No simulator interaction is included (decision time only), matching the
convention of the APEX/AOA-F numbers cited in the table.

Run on the training/eval machine (GPU), from the SearchWorld repo root:

    python deploy/profile_step_latency.py \
        --ckpt /path/to/stage2_epoch15_slim.ckpt --iters 200 --warmup 20

Weights are NOT required for timing (identical architecture/latency);
omit --ckpt to profile a randomly initialised model.
Use --fp16 to also report a half-precision pass (deployment-relevant if the
eval harness runs autocast).

Outputs a human-readable summary and deploy/profile_latency.json with
step_mean_s / step_median_s / step_std_s / episode_s (= 150 x step_mean)
plus per-component timings.
"""

import argparse
import json
import os
import sys
import time

import torch


def build_model(ckpt, device):
    """Build the deployment-configuration model (stage-3 full agent)."""
    # Repo-root imports (must come after sys.path setup in main()).
    from model.searchworld.searchworld import SearchWorld

    model = SearchWorld(
        enable_semantic=False,       # no segmentation at deployment
        enable_rgb_stylegan=False,   # no RGB decoding at deployment
        enable_rgb_diffusion=False,  # no diffusion at deployment
        is_gwm_pretrain=False,       # action policy present
        enable_bev_decoder=True,     # value-map head present
    )
    n_loaded = 0
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        sd = {k[len("model."):]: v for k, v in sd.items()
              if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(unexpected)
        print(f"[ckpt] loaded {n_loaded} params | "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[ckpt] first missing: {missing[:5]}")
    model.to(device).eval()
    return model


def make_step_batch(model, batch_size: int, image_size: int, device: str):
    """Synthetic single-step batch in the deployed format (s=1)."""
    b = batch_size
    batch = {
        "image": torch.rand(b, 1, 3, image_size, image_size, device=device),
        "text_feat": torch.randn(b, 1, 768, device=device),
        "relative_pose": torch.randn(b, 1, 4, device=device),
        "action": torch.zeros(b, 1, dtype=torch.long, device=device),
    }
    # Zero-initialised recurrent state (timing only; values irrelevant).
    h = torch.zeros(b, model.rssm.hidden_state_dim, device=device)
    z = torch.zeros(b, model.rssm.state_dim, device=device)
    batch["history"], batch["sample"] = h, z
    return batch


@torch.inference_mode()
def one_step(model, batch):
    """One full decision step; mutates batch['history'/'sample'] in place."""
    try:
        action_out, path_out, history, sample, sem, rgb, depth, bev = \
            model.inference(batch,
                            enable_semantic=False,
                            enable_rgb=False,
                            enable_depth=False)
    except TypeError:
        # Older signature without the depth flag.
        action_out, path_out, history, sample, sem, rgb, bev = \
            model.inference(batch,
                            enable_semantic=False,
                            enable_rgb=False)
    batch["history"], batch["sample"] = history, sample
    return action_out


@torch.inference_mode()
def time_component(fn, iters: int, warmup: int, sync) -> float:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts = torch.tensor(ts)
    return {"mean": ts.mean().item(),
            "median": ts.median().item(),
            "std": ts.std().item(),
            "min": ts.min().item()}


@torch.inference_mode()
def profile(args, device: str, use_fp16: bool) -> dict:
    from model.searchworld.searchworld import SearchWorld  # noqa: F401

    model = build_model(args.ckpt, device)
    batch = make_step_batch(model, args.batch, args.image_size, device)

    if device.startswith("cuda"):
        def sync():
            torch.cuda.synchronize()
    else:
        def sync():
            pass

    step_fn = lambda: one_step(model, batch)  # noqa: E731
    if use_fp16:
        step_fn = lambda: torch.autocast(device_type="cuda",  # noqa: E731
                                         dtype=torch.float16)(one_step)(
                                             model, batch)

    # Full-step latency.
    step = time_component(step_fn, args.iters, args.warmup, sync)

    # Per-component latency (same batch, manual decomposition).
    obs = time_component(
        lambda: model.observation_encoder(batch), args.iters, args.warmup,
        sync)
    rssm = time_component(
        lambda: model.rssm.observe_step(
            batch["history"], batch["sample"], batch["action"],
            model.observation_encoder(batch)["embedding"][:, -1],
            use_sample=False),
        args.iters, args.warmup, sync)
    # Policy + BEV decoder on a cached state.
    state = torch.cat([batch["history"], batch["sample"]], dim=-1)
    policy = time_component(
        lambda: model.action_policy.inference(state, batch),
        args.iters, args.warmup, sync)
    bev = time_component(lambda: model.bev_decoder(state),
                         args.iters, args.warmup, sync)

    # Optional: multi-view encoder scaling (paper notation V).
    multi_view = {}
    if args.views > 1:
        mv_batch = dict(batch)
        mv_batch["image"] = batch["image"].repeat(
            1, args.views, 1, 1, 1).reshape(
            args.batch * args.views, 1, 3, args.image_size,
            args.image_size)
        mv_batch["text_feat"] = batch["text_feat"].repeat(
            1, args.views, 1).reshape(args.batch * args.views, 1, 768)
        mv_batch["relative_pose"] = batch["relative_pose"].repeat(
            1, args.views, 1).reshape(args.batch * args.views, 1, 4)
        multi_view[f"encoder_x{args.views}_views"] = time_component(
            lambda: model.observation_encoder(mv_batch), args.iters,
            args.warmup, sync)

    result = {
        "device": torch.cuda.get_device_name(0)
        if device.startswith("cuda") else "cpu",
        "fp16": use_fp16,
        "batch": args.batch,
        "image_size": args.image_size,
        "iters": args.iters,
        "step_s": step,
        "episode_s": 150 * step["mean"],
        "components_s": {
            "obs_encoder": obs,
            "rssm_observe_step": rssm,
            "action_policy": policy,
            "bev_decoder": bev,
        },
        **multi_view,
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="Optional checkpoint (weights do not affect latency)")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch", type=int, default=1,
                   help="Agents per step (deployment = 1)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--views", type=int, default=1,
                   help="Also time the encoder on V views per step")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--out", default="deploy/profile_latency.json")
    args = p.parse_args()

    # Ensure repo root is importable regardless of cwd.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    results = [profile(args, args.device, False)]
    if args.fp16 and args.device.startswith("cuda"):
        results.append(profile(args, args.device, True))

    payload = {"results": results}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    for r in results:
        tag = "fp16" if r["fp16"] else "fp32"
        s = r["step_s"]
        print(f"\n=== {r['device']} ({tag}) ===")
        print(f"step: mean {s['mean']*1e3:.1f} ms | median "
              f"{s['median']*1e3:.1f} ms | std {s['std']*1e3:.2f} ms")
        print(f"episode (150 steps): {r['episode_s']:.1f} s")
        for name, c in r["components_s"].items():
            print(f"  {name:20s} {c['mean']*1e3:7.1f} ms")
        for name, c in r.items():
            if name.startswith("encoder_x"):
                print(f"  {name:20s} {c['mean']*1e3:7.1f} ms")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
