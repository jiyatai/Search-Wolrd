"""Open-loop decoder sanity check for the stage-2 expert checkpoint.

Runs the full model (observation encoder -> RSSM -> all four decoder heads)
open-loop on held-out test episodes, saves every decoder's outputs to an .npz,
and prints per-decoder health metrics:

  1. action_policy  - discrete command logits (8-way) vs GT driving_command
  2. rgb_decoder    - StyleGAN RGB imagination vs GT camera image (PSNR/SSIM)
  3. bev_decoder    - BEV map prediction vs GT bev_map (per-channel MSE/SSIM)
  4. rssm           - KL(posterior || prior) health check

Usage (inside the training container, cwd=/workspace/SearchWorld):
  python openloop_decoder_test.py \
      --ckpt ../datasets_shared/WorldSearch_data/deploy_bundle/stage2_epoch15_slim.ckpt \
      --episode-dir ../datasets_shared/WorldSearch_data/BrushifyUrban_parquet_with_bev/test/episode_0091
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

# Make `model.*` importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 1) Register all gin configurables FIRST (same import chain as train.py),
#    then parse the config, otherwise RSSM bindings are silently skipped.
# ---------------------------------------------------------------------------

from model.dataset.uav_dataset import UAVDataModule  # noqa: F401
from model.dataset.uav_parquet_dataset import UAVParquetDataModule  # noqa: F401
from model.trainer import SearchWorldTrainer  # noqa: F401

import gin  # noqa: E402

ACTION_SIZE = 8
# Training resolution for this model (data_constants.INPUT_IMAGE_SIZE).
INPUT_IMAGE_SIZE = (320, 512)


# ---------------------------------------------------------------------------
# Data loading (mirrors UAVParquetDataset._get_element, without the Dataset
# wrapper so we can stream a whole episode as one long sequence).
# ---------------------------------------------------------------------------
def load_episode(episode_dir, text_feat=None):
    pq = os.path.join(episode_dir, 'output_0000.pqt')
    meta_path = os.path.join(episode_dir, 'output_0000_metadata.json')
    df = pd.read_parquet(pq)
    with open(meta_path) as f:
        meta = json.load(f)

    from PIL import Image
    import torchvision.transforms.functional as tvf

    images, poses, bevs, actions = [], [], [], []
    for _, row in df.iterrows():
        img = Image.open(io.BytesIO(row['camera_image']))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.transpose(np.array(img), (2, 0, 1)) / 255.0  # (3,H,W)
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = torch.nn.functional.interpolate(
            img, size=list(INPUT_IMAGE_SIZE),
            mode='bicubic', align_corners=False).squeeze(0)
        images.append(img)
        poses.append(torch.zeros(4, dtype=torch.float32))
        bev = np.array(row['bev_map'], dtype=np.float32)
        bev = bev.reshape(row['bev_map_shape'])  # (3,256,256)
        bevs.append(torch.from_numpy(bev))
        actions.append(int(row['driving_command']))

    episode = {
        'image': torch.stack(images),          # (T,3,224,224)
        'relative_pose': torch.stack(poses),   # (T,4)
        'bev_gt': torch.stack(bevs),           # (T,3,256,256)
        'action': torch.tensor(actions, dtype=torch.int64),  # (T,)
        'task_description': meta['task_description'],
    }
    if text_feat is not None:
        episode['text_feat'] = text_feat  # (768,) broadcast over time below
    return episode


def episode_to_batch(episode, t, history_len=2):
    """Build a model batch for the step ending at time t (inclusive)."""
    h = history_len
    tf = episode['text_feat']  # (768,)
    batch = {
        'image': episode['image'][t - h + 1:t + 1].unsqueeze(0),      # (1,h,3,224,224)
        'relative_pose': episode['relative_pose'][t - h + 1:t + 1].unsqueeze(0),
        'text_feat': tf.unsqueeze(0).unsqueeze(0).expand(1, h, -1).contiguous(),  # (1,h,768)
        # NOTE: in training data bev_memory is the SAME frame's GT BEV map
        # (see uav_parquet_dataset._get_element: bev_memory = bev_gt), so
        # it aligns with the image sequence here. In true closed-loop
        # deployment this must be built online instead.
        'bev_memory': episode['bev_gt'][t - h + 1:t + 1].unsqueeze(0),  # (1,h,3,256,256)
    }
    # The RSSM consumes action[t-1] when producing the transition into t.
    batch['action'] = episode['action'][t - 1:t].unsqueeze(0)         # (1,1)
    return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def ssim(a, b):
    """Simple SSIM for 2-D arrays in [0,1] with an 11x11 uniform window."""
    from scipy.ndimage import uniform_filter
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = uniform_filter(a, 11)
    mu_b = uniform_filter(b, 11)
    var_a = uniform_filter(a * a, 11) - mu_a ** 2
    var_b = uniform_filter(b * b, 11) - mu_b ** 2
    cov = uniform_filter(a * b, 11) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episode-dir', required=True)
    ap.add_argument('--config', default='configs/stage2_expert_parquet_config.gin')
    ap.add_argument('--history-len', type=int, default=2)
    ap.add_argument('--max-steps', type=int, default=48,
                    help='max open-loop steps to evaluate (episode is 51 frames)')
    ap.add_argument('--imagine-steps', type=int, default=16,
                    help='after burn-in, run this many self-supervised '
                         'imagination steps (no GT obs) and score decoders '
                         'against the remaining GT frames')
    ap.add_argument('--imagine-mode', choices=['gt', 'policy'], default='gt',
                    help="'gt': advance latent with GT actions (pure world-"
                         "model dynamics test); 'policy': advance with the "
                         "action_policy's own predictions (approx. closed loop)"
                         )
    ap.add_argument('--text-feat-cache', default=None,
                    help='optional path to text_feat_cache.pt with per-description feats')
    ap.add_argument('--save', default='decoder_test_results.npz')
    ap.add_argument('--save-images', type=int, default=6,
                    help='how many (gt, pred) image/bev pairs to dump into the npz')
    ap.add_argument('--save-vis', default=None,
                    help='directory to write PNG visualisations into '
                         '(defaults to alongside --save)')
    args = ap.parse_args()

    gin.parse_config_file(args.config, skip_unknown=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SearchWorldTrainer.load_from_checkpoint(args.ckpt, strict=False)
    model = model.model.to(device).eval()  # unwrap to SearchWorld
    m = model

    # ------------------------------------------------------------------
    # Text feature: reuse the dataset's cached text features (L2
    # normalised SigLIP2 text embeddings, keyed by task description).
    # ------------------------------------------------------------------
    text_feat = None
    cache_candidates = [
        args.text_feat_cache,
        os.path.join(os.path.dirname(args.episode_dir.rstrip('/')),
                     f'text_feat_cache_{os.path.basename(args.episode_dir.rstrip("/"))}.pt'),
        os.path.join(os.path.dirname(os.path.dirname(args.episode_dir.rstrip('/'))),
                     'text_feat_cache_test.pt'),
        '/workspace/datasets_shared/WorldSearch_data/expert_parquet/text_feat_cache_test.pt',
    ]
    meta_path = os.path.join(args.episode_dir, 'output_0000_metadata.json')
    with open(meta_path) as f:
        task_desc = json.load(f).get('task_description')
    for cache_path in cache_candidates:
        if cache_path and os.path.exists(cache_path):
            cache = torch.load(cache_path).get('text_feat', {})
            if task_desc in cache:
                text_feat = cache[task_desc].float()
                if text_feat.dim() > 1:
                    text_feat = text_feat.reshape(-1)
                print(f'[text] loaded cached feat for "{task_desc[:50]}..." '
                      f'from {cache_path}')
                break
            elif len(cache) > 0:
                # Episode metadata carries the generic prompt, but the cache is
                # keyed by the specific target-object descriptions. All cached
                # descriptions belong to the same "Search for target object"
                # task; use the first one so the text branch sees a realistic
                # (L2-normalised) embedding.
                first_key = sorted(cache.keys())[0]
                text_feat = cache[first_key].float()
                if text_feat.dim() > 1:
                    text_feat = text_feat.reshape(-1)
                print(f'[text] episode desc "{task_desc}" not in cache; using '
                      f'first cached desc "{first_key[:40]}..." from {cache_path}')
                break
    if text_feat is None:
        print('[text] no cache entry found, falling back to zeros(768) '
              '(still a valid decoder smoke test)')
        text_feat = torch.zeros(768)

    episode = load_episode(args.episode_dir, text_feat)
    T = episode['image'].shape[0]
    # Keep t in [h, h+steps) with t <= T-1 and t-h+1 >= 0 so every
    # history slice [t-h+1:t+1] is exactly h frames long. Reserve the
    # tail frames for the imagination rollout if requested.
    n_imag = max(0, min(args.imagine_steps, T - 1 - args.history_len))
    steps = min(args.max_steps, T - args.history_len - n_imag)
    steps = max(0, steps)
    print(f'[data] episode={os.path.basename(args.episode_dir)} frames={T} '
          f'desc="{episode["task_description"]}" steps={steps}')

    # ------------------------------------------------------------------
    # Open-loop rollout: teacher-forced observations, one-step-ahead
    # predictions from every decoder at each t.
    # ------------------------------------------------------------------
    results = {
        'action_pred': [], 'action_gt': [], 'action_softmax_max': [],
        'rgb_psnr': [], 'rgb_ssim': [], 'bev_mse_ch': [], 'bev_ssim_ch': [],
        'kl_post_prior': [], 'path_pred': [],
    }
    dump = {'rgb_gt': [], 'rgb_pred': [], 'bev_gt': [], 'bev_pred': []}

    h0 = torch.zeros(1, m.rssm.hidden_state_dim, device=device)
    s0 = torch.zeros(1, m.rssm.state_dim, device=device)

    with torch.no_grad():
        for t in range(args.history_len, args.history_len + steps):
            batch = episode_to_batch(episode, t, args.history_len)
            batch = {k: v.to(device) for k, v in batch.items()}

            # --- 1) observation encoder + RSSM posterior update ---
            obs = m.observation_encoder(batch)
            post = m.rssm.observe_step(
                h0, s0, batch['action'].squeeze(1),
                obs['embedding'][:, -1], use_sample=False)['posterior']

            # KL(posterior || prior) for RSSM health.
            prior = m.rssm.imagine_step(h0, s0, batch['action'].squeeze(1),
                                        use_sample=False)
            mu_p, sig_p = prior['mu'], prior['sigma']
            mu_q, sig_q = post['mu'], post['sigma']
            kl = 0.5 * (
                2 * torch.log(sig_p / sig_q) - 1
                + (sig_q / sig_p) ** 2 + ((mu_p - mu_q) / sig_p) ** 2
            ).sum(-1)
            results['kl_post_prior'].append(kl.mean().item())

            h0 = post['hidden_state'].detach()
            s0 = post['sample'].detach()

            state = torch.cat([h0, s0], dim=-1)  # (1, state_dim)

            # --- 2) action policy (single-step interface: batch with s=1) ---
            step_batch = {
                'image': batch['image'][:, -1:],
                'relative_pose': batch['relative_pose'][:, -1:],
                'text_feat': batch['text_feat'][:, -1:],   # (1,1,768)
                'bev_memory': batch['bev_memory'][:, -1:],
            }
            action_logits, path_out = m.action_policy.inference(state, step_batch)
            # action_logits: (1,1,8) -> (8,)
            logits = action_logits.squeeze().cpu().numpy()
            pred = int(logits.argmax())
            gt = int(episode['action'][t].item())
            softmax = np.exp(logits - logits.max())
            softmax = softmax / softmax.sum()
            results['action_pred'].append(pred)
            results['action_gt'].append(gt)
            results['action_softmax_max'].append(float(softmax.max()))
            if path_out is not None:
                results['path_pred'].append(
                    path_out.squeeze().cpu().numpy().copy())

            # --- 3) RGB decoder (StyleGAN) ---
            rgb_out = m.rgb_decoder(state)  # dict rgb_1/2/4
            rgb_pred = rgb_out['rgb_1'].squeeze(0).cpu().numpy()  # (3,320,512)
            rgb_gt = episode['image'][t].numpy()
            results['rgb_psnr'].append(psnr(rgb_pred, rgb_gt))
            results['rgb_ssim'].append(
                float(np.mean([ssim(rgb_pred[c], rgb_gt[c]) for c in range(3)])))

            # --- 4) BEV decoder ---
            bev_pred = m.bev_decoder(state).squeeze(0).cpu().numpy()  # (3,256,256)
            bev_gt = episode['bev_gt'][t].numpy()
            mse_ch = ((bev_pred - bev_gt) ** 2).mean(axis=(1, 2))
            results['bev_mse_ch'].append(mse_ch)
            results['bev_ssim_ch'].append(
                [ssim(bev_pred[i], bev_gt[i]) for i in range(3)])

            if len(dump['rgb_gt']) < args.save_images:
                dump['rgb_gt'].append(rgb_gt)
                dump['rgb_pred'].append(np.clip(rgb_pred, 0, 1))
                dump['bev_gt'].append(bev_gt)
                dump['bev_pred'].append(bev_pred)

    # ------------------------------------------------------------------
    # Multi-step imagination rollout: from the final teacher-forced
    # state, let the RSSM advance by its OWN predicted actions (no GT
    # observation ever corrects it) and score every decoder against
    # the remaining GT frames. This is the "walk several more steps"
    # world-model quality test.
    # ------------------------------------------------------------------
    t0 = args.history_len + steps          # first GT frame to compare
    n_imag = max(0, min(n_imag, T - t0))
    imag_results = None
    if n_imag > 0:
        print(f'\n[imagine] rolling out {n_imag} self-supervised steps '
              f'({args.imagine_mode}-action mode, burn-in {steps} '
              f'teacher-forced steps, comparing to GT frames '
              f'{t0}..{t0 + n_imag - 1})')
        with torch.no_grad():
            imag = {
                'action_pred': [], 'action_gt': [], 'action_softmax_max': [],
                'rgb_psnr': [], 'rgb_ssim': [], 'bev_mse_ch': [],
                'bev_ssim_ch': [], 'path_pred': [],
            }
            imag_dump = {'rgb_gt': [], 'rgb_pred': [],
                         'bev_gt': [], 'bev_pred': []}
            # Single-step observation batch for action_policy (text_feat is
            # constant, so burn-in's last step batch is still valid).
            h_i, s_i = h0, s0
            for k in range(n_imag):
                t = t0 + k
                # 1) policy picks the action from the IMAGINED state
                state_i = torch.cat([h_i, s_i], dim=-1)
                step_batch = {
                    'image': batch['image'][:, -1:],
                    'relative_pose': batch['relative_pose'][:, -1:],
                    'text_feat': batch['text_feat'][:, -1:],
                    'bev_memory': batch['bev_memory'][:, -1:],
                }
                action_logits, path_out = m.action_policy.inference(
                    state_i, step_batch)
                logits = action_logits.squeeze().cpu().numpy()
                pred = int(logits.argmax())
                softmax = np.exp(logits - logits.max())
                softmax = softmax / softmax.sum()
                gt = int(episode['action'][t].item())
                imag['action_pred'].append(pred)
                imag['action_gt'].append(gt)
                imag['action_softmax_max'].append(float(softmax.max()))
                if path_out is not None:
                    imag['path_pred'].append(
                        path_out.squeeze().cpu().numpy().copy())

                # 2) advance the latent — no GT observation is ever seen.
                #    'gt' mode: GT action (pure world-model dynamics).
                #    'policy' mode: the policy's own predicted action.
                if args.imagine_mode == 'gt':
                    a_i = torch.tensor([gt], device=device)
                else:
                    a_i = torch.tensor([pred], device=device)
                prior = m.rssm.imagine_step(h_i, s_i, a_i, use_sample=False)
                h_i = prior['hidden_state']
                s_i = prior['sample']

                # 3) decode the imagined state and compare to GT frame t
                state_i = torch.cat([h_i, s_i], dim=-1)
                rgb_pred = m.rgb_decoder(state_i)['rgb_1'].squeeze(0) \
                    .cpu().numpy()
                rgb_gt = episode['image'][t].numpy()
                imag['rgb_psnr'].append(psnr(rgb_pred, rgb_gt))
                imag['rgb_ssim'].append(
                    float(np.mean([ssim(rgb_pred[c], rgb_gt[c])
                                   for c in range(3)])))
                bev_pred = m.bev_decoder(state_i).squeeze(0).cpu().numpy()
                bev_gt = episode['bev_gt'][t].numpy()
                mse_ch = ((bev_pred - bev_gt) ** 2).mean(axis=(1, 2))
                imag['bev_mse_ch'].append(mse_ch)
                imag['bev_ssim_ch'].append(
                    [ssim(bev_pred[i], bev_gt[i]) for i in range(3)])
                if len(imag_dump['rgb_gt']) < args.save_images:
                    imag_dump['rgb_gt'].append(rgb_gt)
                    imag_dump['rgb_pred'].append(np.clip(rgb_pred, 0, 1))
                    imag_dump['bev_gt'].append(bev_gt)
                    imag_dump['bev_pred'].append(bev_pred)

            print(f'[imagine/action]  acc='
                  f'{np.mean(np.array(imag["action_pred"]) == np.array(imag["action_gt"])):.3f}'
                  f'  softmax max={np.mean(imag["action_softmax_max"]):.3f}')
            print(f'[imagine/rgb]     PSNR={np.mean(imag["rgb_psnr"]):.2f}dB  '
                  f'SSIM={np.mean(imag["rgb_ssim"]):.3f}')
            bev_ssim_imag = np.stack(imag['bev_ssim_ch'])
            for i, name in enumerate(['exploration', 'obstacle', 'value']):
                print(f'[imagine/bev/{name}] SSIM='
                      f'{bev_ssim_imag[:, i].mean():.3f}')
            # per-step degradation curve: how fast does RGB drift?
            psnrs = np.array(imag['rgb_psnr'])
            chunk = max(1, n_imag // 4)
            curve = [f'{psnrs[i:i+chunk].mean():.1f}' for i in
                     range(0, n_imag, chunk)]
            print(f'[imagine/drift]   RGB PSNR by quarter: {" -> ".join(curve)}'
                  f'  (teacher-forced ref: {np.mean(results["rgb_psnr"]):.1f}dB)')
            imag_results = {k: np.stack(v) for k, v in imag.items()
                            if len(v) > 0}
            if args.save_images > 0 and imag_dump['rgb_gt']:
                imag_results['rgb_gt'] = np.stack(imag_dump['rgb_gt'])
                imag_results['rgb_pred'] = np.stack(imag_dump['rgb_pred'])
                imag_results['bev_gt'] = np.stack(imag_dump['bev_gt'])
                imag_results['bev_pred'] = np.stack(imag_dump['bev_pred'])

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    action_pred = np.array(results['action_pred'])
    action_gt = np.array(results['action_gt'])
    acc = float((action_pred == action_gt).mean())
    print('\n================ Decoder health report ================')
    print(f'[action_policy] acc={acc:.3f} ({(action_pred == action_gt).sum()}'
          f'/{len(action_gt)})  '
          f'mean softmax max={np.mean(results["action_softmax_max"]):.3f}')
    print(f'  pred dist: {np.bincount(action_pred, minlength=ACTION_SIZE).tolist()}')
    print(f'  gt   dist: {np.bincount(action_gt, minlength=ACTION_SIZE).tolist()}')

    print(f'[rgb_decoder]   PSNR={np.mean(results["rgb_psnr"]):.2f}dB  '
          f'SSIM={np.mean(results["rgb_ssim"]):.3f}  (n={len(results["rgb_psnr"])})')

    bev_mse = np.stack(results['bev_mse_ch'])  # (n,3)
    bev_ssim = np.stack(results['bev_ssim_ch'])
    ch_names = ['exploration', 'obstacle', 'value']
    for i, name in enumerate(ch_names):
        print(f'[bev_decoder/{name}] MSE={bev_mse[:, i].mean():.4f}  '
              f'SSIM={bev_ssim[:, i].mean():.3f}')

    print(f'[rssm]          mean KL(post||prior)='
          f'{np.mean(results["kl_post_prior"]):.3f}')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = {
        'action_pred': action_pred,
        'action_gt': action_gt,
        'action_softmax_max': np.array(results['action_softmax_max']),
        'rgb_psnr': np.array(results['rgb_psnr']),
        'rgb_ssim': np.array(results['rgb_ssim']),
        'bev_mse_ch': bev_mse,
        'bev_ssim_ch': bev_ssim,
        'kl_post_prior': np.array(results['kl_post_prior']),
        'episode_dir': args.episode_dir,
    }
    if len(results['path_pred']):
        out['path_pred'] = np.stack(results['path_pred'])
    if args.save_images > 0:
        out['rgb_gt'] = np.stack(dump['rgb_gt'])
        out['rgb_pred'] = np.stack(dump['rgb_pred'])
        out['bev_gt'] = np.stack(dump['bev_gt'])
        out['bev_pred'] = np.stack(dump['bev_pred'])
    if imag_results is not None:
        for k, v in imag_results.items():
            out[f'imagine_{k}'] = v
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    np.savez_compressed(args.save, **out)
    print(f'\n[saved] {args.save}')
    for k, v in out.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: {v.shape} {v.dtype}')
        else:
            print(f'  {k}: {v}')

    # ------------------------------------------------------------------
    # PNG visualisations: (gt | pred) side-by-side strips for the
    # teacher-forced section and every dumped imagination step.
    # ------------------------------------------------------------------
    vis_dir = args.save_vis or os.path.dirname(os.path.abspath(args.save))
    os.makedirs(vis_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.save))[0]

    def save_strip(gt, pred, title, path, cmap_ch=None):
        """gt/pred are (3,H,W) or a channel (H,W); writes gt|pred png."""
        from PIL import Image
        panels = []
        for arr in (gt, pred):
            a = np.clip(arr, 0, 1)
            if a.ndim == 3:            # (3,H,W) RGB
                panels.append((np.transpose(a, (1, 2, 0)) * 255)
                              .astype(np.uint8))
            else:                       # single channel -> grayscale
                if cmap_ch == 'jet':
                    # cheap jet-ish colormap for BEV value maps
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.cm as cm
                    rgba = cm.jet(a)[..., :3]
                    panels.append((rgba * 255).astype(np.uint8))
                else:
                    panels.append((np.stack([a] * 3, -1) * 255)
                                  .astype(np.uint8))
        w = max(p.shape[1] for p in panels)
        strip = np.concatenate(
            [np.pad(p, ((0, 0), (0, w - p.shape[1]), (0, 0))) for p in panels],
            axis=1)
        Image.fromarray(strip).save(path)
        print(f'[vis] {title} -> {path}')

    for phase, prefix in (('tf', ''), ('imag', 'imagine_')):
        rgb_gt_k, rgb_pred_k = f'{prefix}rgb_gt', f'{prefix}rgb_pred'
        if rgb_gt_k not in out:
            continue
        tag = stem if phase == 'tf' else f'{stem}_imag'
        n = out[rgb_gt_k].shape[0]
        for i in range(n):
            save_strip(out[rgb_gt_k][i], out[rgb_pred_k][i],
                       f'{tag} rgb[{i}]',
                       os.path.join(vis_dir, f'{tag}_rgb_{i}.png'))
            # BEV: stack the 3 channels horizontally, gt on top of pred
            bev_pair = []
            for arr in (out[f'{prefix}bev_gt'][i], out[f'{prefix}bev_pred'][i]):
                row = np.concatenate(
                    [arr[c] for c in range(arr.shape[0])], axis=1)
                bev_pair.append((np.clip(row, 0, 1) * 255).astype(np.uint8))
            from PIL import Image
            bev_img = np.concatenate(bev_pair, axis=0)  # gt row, pred row
            Image.fromarray(bev_img).save(
                os.path.join(vis_dir, f'{tag}_bev_{i}.png'))
            print(f'[vis] {tag} bev[{i}] -> '
                  f'{os.path.join(vis_dir, f"{tag}_bev_{i}.png")}')


if __name__ == '__main__':
    main()
