"""
eval_effective_rank.py — metric collapse nội tại (RankMe) cho checkpoint I-JEPA.

Script ĐỘC LẬP (không cắm vào training loop để tránh phức tạp DDP). Đo mức độ
"phủ" của không gian đặc trưng: representation collapse -> effective rank sụp.

RankMe (Garrido et al., ICML 2023):
  cho ma trận feature Z (N, D), lấy singular values sigma qua SVD,
  p = sigma / (sum sigma) + 1e-7 ; effective_rank = exp( -sum p*log p ).

Cách dùng:
  # 1 checkpoint
  python eval_effective_rank.py --ckpt logs/mri_vit_base_R1/mri_vit_base_R1-ep50.pth.tar \
      --data_root /network-volume/hungph/data/SAG_3D_DESS_v2_full

  # nhiều checkpoint (glob) -> xuất CSV rank_vs_epoch.csv để vẽ figure cơ chế
  python eval_effective_rank.py --ckpt "logs/mri_vit_base_R1/*-ep*.pth.tar" \
      --data_root /network-volume/hungph/data/SAG_3D_DESS_v2_full --out rank_vs_epoch.csv

Kỳ vọng: R0 rank crash sớm rồi hồi; R1 giữ rank cao xuyên suốt.
"""

import os
import re
import csv
import glob
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models import vision_transformer as vit
from src.transforms import make_transforms
from downstream import DownstreamDataset


def rankme(features, eps=1e-7):
    """RankMe effective rank cho features (N, D)."""
    feats = features.float()
    # SVD (dùng float64 trên CPU cho ổn định số học)
    s = torch.linalg.svdvals(feats.double())
    s = s[s > 0]
    p = s / s.sum() + eps
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())


def load_encoder(ckpt_path, device, crop_size, patch_size):
    encoder = vit.vit_base(img_size=[list(crop_size)], patch_size=tuple(patch_size))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # ưu tiên target_encoder, fallback encoder
    state = ckpt.get("target_encoder", ckpt.get("encoder", {}))
    new_state = {}
    for k, v in state.items():
        new_state[k[7:] if k.startswith("module.") else k] = v
    msg = encoder.load_state_dict(new_state, strict=False)
    print(f"  [load] {os.path.basename(ckpt_path)} msg: {msg}")
    encoder.to(device).eval()
    return encoder


@torch.no_grad()
def extract_features(encoder, loader, device, n_samples):
    feats = []
    seen = 0
    for images, _ in loader:
        images = images.to(device)
        out = encoder(images, masks=None)      # (B, num_patches, D)
        pooled = out.float().mean(dim=1)        # mean-pool theo token -> (B, D)
        feats.append(pooled.cpu())
        seen += pooled.shape[0]
        if seen >= n_samples:
            break
    feats = torch.cat(feats, dim=0)[:n_samples]
    return feats


def epoch_of(path):
    m = re.search(r"ep(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Đường dẫn 1 checkpoint HOẶC glob pattern nhiều checkpoint.")
    parser.add_argument("--data_root", type=str,
                        default="/network-volume/hungph/data/SAG_3D_DESS_v2_full")
    parser.add_argument("--mri_folder", type=str, default=None,
                        help="Mặc định = <data_root>/MRI_Numpy")
    parser.add_argument("--split_csv", type=str, default="validation.csv",
                        help="Split held-out để đo (mặc định validation.csv).")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--crop_size", type=int, nargs=3, default=[120, 160, 160])
    parser.add_argument("--patch_size", type=int, nargs=3, default=[12, 16, 16])
    parser.add_argument("--fg_method", type=str, default="percentile")
    parser.add_argument("--p_bg", type=int, default=20)
    parser.add_argument("--normalize", type=str, default="foreground_zscore",
                        help="Phải KHỚP normalization dùng khi pretrain checkpoint.")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV xuất rank_vs_epoch (nếu nhiều checkpoint).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mri_folder = args.mri_folder or os.path.join(args.data_root, "MRI_Numpy")

    # eval-transform: KHÔNG RRC, chỉ normalization (xác định)
    eval_transform = make_transforms(training=False, normalize=args.normalize,
                                     fg_method=args.fg_method, p_bg=args.p_bg)
    df = pd.read_csv(os.path.join(args.data_root, args.split_csv))
    dataset = DownstreamDataset(df, mri_folder, mri_transforms=eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    # danh sách checkpoint
    ckpts = sorted(glob.glob(args.ckpt), key=epoch_of) if any(c in args.ckpt for c in "*?[") \
        else [args.ckpt]
    ckpts = [c for c in ckpts if os.path.exists(c)]
    if not ckpts:
        raise FileNotFoundError(f"Không tìm thấy checkpoint khớp: {args.ckpt}")

    rows = []
    for ck in ckpts:
        encoder = load_encoder(ck, device, args.crop_size, args.patch_size)
        feats = extract_features(encoder, loader, device, args.n_samples)
        er = rankme(feats)
        fstd = float(feats.std(dim=0).mean().item())
        ep = epoch_of(ck)
        print(f"[rank] {os.path.basename(ck)} | epoch={ep} | N={feats.shape[0]} "
              f"D={feats.shape[1]} | effective_rank={er:.2f} | feat_std={fstd:.4f}")
        rows.append({"checkpoint": os.path.basename(ck), "epoch": ep,
                     "effective_rank": er, "feat_std": fstd,
                     "n_samples": feats.shape[0], "feat_dim": feats.shape[1]})

    if args.out and len(rows) >= 1:
        rows_sorted = sorted(rows, key=lambda r: r["epoch"])
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            w.writeheader()
            w.writerows(rows_sorted)
        print(f"\nSaved rank-vs-epoch CSV to {args.out}")


if __name__ == "__main__":
    main()
