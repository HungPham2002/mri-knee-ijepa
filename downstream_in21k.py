"""
Downstream benchmark cho backbone khởi tạo từ ImageNet-21k (inflated 2D->3D).

Mục tiêu: fair benchmark với backbone pretrained bằng I-JEPA. Script này dùng
CHÍNH XÁC cùng protocol (dataset, transforms, model head, optimizer, scheduler,
loss, training loop, eval) như `downstream.py` -- toàn bộ phần đó được tái sử dụng
qua `run_downstream_experiment` và các helper import từ `downstream.py`. Khác biệt
DUY NHẤT là nguồn trọng số của encoder:

  - `downstream.py`      : load checkpoint I-JEPA (target_encoder/encoder).
  - `downstream_in21k.py`: load trọng số ViT-B/16 ImageNet-21k (timm), inflate
                           2D->3D bằng `VisionTransformer.load_pretrain`.

Trọng số ImageNet-21k được lấy qua timm (`vit_base_patch16_224.orig_in21k`). Lần
đầu sẽ download, chuẩn hoá (bỏ cls token khỏi pos_embed để khớp với logic inflate
trong `src/models/vision_transformer.py`) rồi lưu ra đĩa. Các lần sau nạp lại từ
file đã lưu, không download lại.
"""

import os
import argparse

import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.models import vision_transformer as vit
from src.transforms import make_transforms

# Tái sử dụng nguyên vẹn protocol downstream để đảm bảo identical benchmark.
from downstream import (
    DownstreamDataset,
    ViTClassifier,
    set_requires_grad,
    run_downstream_experiment,
    add_common_downstream_args,
    build_encoder,
)


def prepare_in21k_weights(cache_path: str, timm_model: str) -> str:
    """Trả về đường dẫn tới file trọng số ImageNet-21k đã chuẩn hoá (vẫn là 2D).

    Nếu file đã tồn tại -> dùng lại (không download). Nếu chưa -> dùng timm để
    download trọng số pretrained, chuẩn hoá rồi lưu ra `cache_path` cho lần sau.

    Chuẩn hoá: bỏ `cls_token` và cắt hàng cls khỏi `pos_embed`
    ([1, 197, 768] -> [1, 196, 768]) để khớp với `load_pretrain`, vốn coi toàn bộ
    pos_embed là grid patch thuần (14x14) rồi nội suy 3D sang grid của volume.
    Việc inflate 2D->3D (mean-kernel cho patch conv, trilinear cho pos_embed) do
    `VisionTransformer.load_pretrain` đảm nhiệm khi nạp.
    """
    if os.path.exists(cache_path):
        print(f"=> Found cached ImageNet-21k weights: {cache_path}")
        return cache_path

    print(f"=> Cached weights not found. Downloading '{timm_model}' via timm...")
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            "Cần cài timm để download trọng số ImageNet-21k: `pip install timm`"
        ) from e

    # num_classes=0 -> chỉ lấy backbone (không có head/pre_logits thừa).
    model = timm.create_model(timm_model, pretrained=True, num_classes=0)
    state_dict = model.state_dict()

    # Chuẩn hoá pos_embed: bỏ hàng cls token (vị trí 0) nếu có.
    if "pos_embed" in state_dict:
        pos_embed = state_dict["pos_embed"]
        num_patch_tokens = model.patch_embed.num_patches  # 196 cho patch16/224
        if pos_embed.shape[1] == num_patch_tokens + 1:
            state_dict["pos_embed"] = pos_embed[:, 1:, :].contiguous()
    state_dict.pop("cls_token", None)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(state_dict, cache_path)
    print(f"=> Saved normalized ImageNet-21k weights to: {cache_path}")
    return cache_path


def main():
    parser = argparse.ArgumentParser()
    add_common_downstream_args(parser)
    parser.add_argument("--timm_model", type=str, default="vit_base_patch16_224.orig_in21k",
                        help="Tên model timm dùng làm nguồn trọng số ImageNet-21k.")
    parser.add_argument("--weights_cache", type=str,
                        default="/home/ubuntu/ecd_hungpham/mri-knee-ijepa/pretrained/vit_base_patch16_224.orig_in21k.pth",
                        help="Nơi lưu/nạp trọng số ImageNet-21k đã chuẩn hoá (2D).")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Bỏ qua mọi pretrained: train from scratch. Tự động ép "
                             "cấu hình full finetuning (--strategy=full).")
    parser.add_argument("--output_dir", type=str, default="output/downstream_in21k")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 1. Prepare Datasets (identical với downstream.py)
    print("Loading dataframes...")
    train_df = pd.read_csv(os.path.join(args.data_root, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_root, "validation.csv"))
    test_df = pd.read_csv(os.path.join(args.data_root, "test.csv"))

    # R1 §4.6 — parity: normalization-only cho cả train và eval (xem chú thích trong downstream.py).
    train_transform = make_transforms(training=False)
    eval_transform = make_transforms(training=False)

    train_dataset = DownstreamDataset(train_df, args.mri_folder, mri_transforms=train_transform)
    val_dataset = DownstreamDataset(val_df, args.mri_folder, mri_transforms=eval_transform)
    test_dataset = DownstreamDataset(test_df, args.mri_folder, mri_transforms=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Build Model (cùng kiến trúc encoder như downstream.py)
    # From scratch bắt buộc full finetuning để encoder được học (đặt trước khi
    # build_encoder để drop_path được chọn đúng theo strategy).
    if args.from_scratch and args.strategy != "full":
        print(f"=> Overriding --strategy '{args.strategy}' -> 'full' for from-scratch training.")
        args.strategy = "full"

    print("Building ViT model...")
    encoder = build_encoder(args)

    if args.from_scratch:
        # Không pretrained: giữ nguyên khởi tạo ngẫu nhiên của encoder.
        print("=> [Init: from scratch] Skipping pretrained weights (random init).")
    else:
        # Nạp trọng số ImageNet-21k (download+cache nếu cần) rồi inflate 2D->3D.
        weights_path = prepare_in21k_weights(args.weights_cache, args.timm_model)
        encoder.load_pretrain(weights_path)
        print(f"Initialized encoder from ImageNet-21k weights: {weights_path}")

    model = ViTClassifier(encoder, num_classes=5)

    # 3. Setup Strategy (identical)
    set_requires_grad(model, args.strategy, args.unfreeze_last_n)
    model.to(device)

    # 4-6. Run the shared training + evaluation protocol (identical với downstream.py)
    run_downstream_experiment(model, train_loader, val_loader, test_loader, device, args)


if __name__ == "__main__":
    main()
