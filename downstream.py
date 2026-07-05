import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from focal_loss.focal_loss import FocalLoss
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
import matplotlib
matplotlib.use("Agg")  # headless / no display on server
import matplotlib.pyplot as plt
import csv
import glob
import json
import re

from src.models import vision_transformer as vit
from src.transforms import make_transforms


class DownstreamDataset(Dataset):
    def __init__(self, df, mri_root, mri_transforms=None):
        self.df = df
        self.mri_root = mri_root
        self.mri_transforms = mri_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # Xử lý tên biến trong dataframe, nếu là file .npz
        mri_path = os.path.join(self.mri_root, str(row.get("mri_path", row.name)))
        
        try:
            npz_data = np.load(mri_path)
            if "data" in npz_data:
                mri_data = npz_data["data"]
            else:
                mri_data = npz_data[npz_data.files[0]]
        except Exception:
            raise FileNotFoundError(f"Could not load MRI data from {mri_path}")
            
        # Dataset pretraining định dạng input là (1, D, H, W)
        mri_data = np.expand_dims(mri_data, 0)

        # Transform nếu có
        if self.mri_transforms:
            mri_data = self.mri_transforms(mri_data)
        
        mri_tensor = torch.tensor(mri_data, dtype=torch.float32)
        label = int(row["kl_grade"])
        return mri_tensor, label


class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)
        
    def forward(self, x):
        # x shape: (B, 1, 120, 160, 160)
        # Bỏ qua params masks
        x = self.encoder(x, masks=None) # (B, N, D)
        # Global Average Pooling theo token dimension
        x = x.mean(dim=1) # (B, D)
        return self.head(x)


def set_requires_grad(model, strategy, unfreeze_last_n=1):
    """
    Hỗ trợ 2 strategy: 
    - 'linear_probe': đóng băng toàn bộ backbone, chỉ train head.
    - 'partial': unfreeze last N layers + layer norm của ViT.
    """
    if strategy == "linear_probe":
        print("=> [Strategy: Linear Probe] Freezing entire encoder...")
        for param in model.encoder.parameters():
            param.requires_grad = False
            
    elif strategy == "partial":
        print(f"=> [Strategy: Partial] Freezing encoder except last {unfreeze_last_n} blocks...")
        for param in model.encoder.parameters():
            param.requires_grad = False
            
        # Unfreeze last N blocks
        for blk in model.encoder.blocks[-unfreeze_last_n:]:
            for param in blk.parameters():
                param.requires_grad = True
                
        # Unfreeze layer norm cuối (nếu có)
        if hasattr(model.encoder, 'norm') and model.encoder.norm is not None:
            for param in model.encoder.norm.parameters():
                param.requires_grad = True
                
    elif strategy == "full":
        print("=> [Strategy: Full] Unfreezing entire model...")
        for param in model.parameters():
            param.requires_grad = True
            
    # Luôn luôn train classification head
    for param in model.head.parameters():
        param.requires_grad = True


def prune_checkpoints(ckpt_dir, keep_last_n):
    """Giữ lại keep_last_n checkpoint mới nhất (theo epoch), xóa phần còn lại."""
    ckpts = glob.glob(os.path.join(ckpt_dir, "ckpt_epoch*.pth"))

    def epoch_of(p):
        m = re.search(r"ckpt_epoch(\d+)\.pth", os.path.basename(p))
        return int(m.group(1)) if m else -1

    ckpts = sorted(ckpts, key=epoch_of)
    for p in ckpts[:-keep_last_n]:
        try:
            os.remove(p)
        except OSError:
            pass


@torch.no_grad()
def evaluate(model, loader, device, desc):
    model.eval()
    preds, targets = [], []
    for images, labels in tqdm(loader, desc=desc):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())
    return {
        "acc": accuracy_score(targets, preds),
        "bacc": balanced_accuracy_score(targets, preds),
        "qwk": cohen_kappa_score(targets, preds, weights="quadratic"),
    }

def add_common_downstream_args(parser):
    """Các argument dùng CHUNG cho mọi script downstream (I-JEPA / ImageNet / ctx).

    Giữ ở một nơi để đảm bảo identical protocol giữa các backbone. Mỗi script chỉ
    thêm phần riêng của nó (đường dẫn checkpoint / output_dir ...).
    """
    parser.add_argument("--data_root", type=str, default="/home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full")
    parser.add_argument("--mri_folder", type=str, default="/home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full/MRI_Numpy")
    parser.add_argument("--strategy", type=str, choices=["linear_probe", "partial", "full"], default="linear_probe",
                        help="Tùy chọn fine-tune.")
    parser.add_argument("--unfreeze_last_n", type=int, default=4, help="Số block cuối của ViT cần unfreeze (nếu strategy=partial).")
    parser.add_argument("--save_after_epoch", type=int, default=10,
                        help="Bắt đầu lưu checkpoint cải thiện từ epoch này (1-indexed).")
    parser.add_argument("--keep_last_n", type=int, default=5,
                        help="Số checkpoint gần nhất giữ lại.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    # --- Recipe finetuning ViT pretrained (mấu chốt để full-FT không sụp) ---
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak LR: áp cho head + block trên cùng; block dưới giảm dần theo layer_decay.")
    parser.add_argument("--layer_decay", type=float, default=0.75,
                        help="Layer-wise LR decay kiểu MAE. 1.0 = tắt (mọi layer cùng LR).")
    parser.add_argument("--warmup_epochs", type=float, default=5.0,
                        help="Số epoch warmup tuyến tính trước khi cosine decay.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="LR cuối của cosine schedule.")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--drop_path", type=float, default=0.1,
                        help="Stochastic depth cho encoder. Tự động tắt (0.0) khi strategy=linear_probe.")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser


def build_param_groups_lrd(model, base_lr, weight_decay, layer_decay):
    """Layer-wise LR decay (MAE-style) cho ViTClassifier(encoder + head).

    - Chỉ gồm param có requires_grad=True (tôn trọng strategy freeze/unfreeze).
    - Block càng gần input -> LR càng nhỏ (scale = layer_decay ** (num_layers - id)),
      giữ nguyên feature pretrained ở tầng thấp; head + LN cuối -> LR = base_lr.
    - Không áp weight decay cho bias / LayerNorm / pos_embed (param 1 chiều).
    """
    encoder = model.encoder
    num_layers = len(encoder.blocks) + 1  # các block + nhóm trên cùng (norm cuối + head)

    def get_layer_id(name):
        if name.startswith("encoder."):
            sub = name[len("encoder."):]
            if sub.startswith("patch_embed") or sub in ("pos_embed", "cls_token"):
                return 0
            if sub.startswith("blocks."):
                return int(sub.split(".")[1]) + 1
            return num_layers  # encoder.norm (LayerNorm cuối)
        return num_layers      # head

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id(name)
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name
        gkey = (layer_id, no_decay)
        if gkey not in groups:
            scale = layer_decay ** (num_layers - layer_id)
            groups[gkey] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[gkey]["params"].append(param)
    return list(groups.values())


def build_scheduler(optimizer, args, steps_per_epoch):
    """Warmup tuyến tính -> cosine decay xuống min_lr (step theo từng iteration).

    Mỗi param-group được scale từ base_lr riêng của nó, nên layer-wise LR decay
    và schedule kết hợp đúng (early layers cũng warmup/cosine theo tỉ lệ của mình).
    """
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.min_lr)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)


def build_encoder(args):
    """Dựng encoder ViT-B/16 3D (cùng cấu hình với pretraining).

    Stochastic depth tự tắt cho linear_probe (encoder đóng băng -> tránh nhiễu
    ngẫu nhiên vào feature dùng cho head).
    """
    drop_path = 0.0 if args.strategy == "linear_probe" else args.drop_path
    return vit.vit_base(img_size=[120, 160, 160], patch_size=(12, 16, 16), drop_path_rate=drop_path)


def main():
    parser = argparse.ArgumentParser()
    add_common_downstream_args(parser)
    parser.add_argument("--ckpt_path", type=str, default="/home/ubuntu/ecd_hungpham/mri-knee-ijepa/output/mri_vit_base_300ep-ep100.pth.tar")
    parser.add_argument("--output_dir", type=str, default="output/downstream")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # 1. Prepare Datasets
    print("Loading dataframes...")
    train_df = pd.read_csv(os.path.join(args.data_root, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_root, "validation.csv"))
    test_df = pd.read_csv(os.path.join(args.data_root, "test.csv"))
    
    # Init transforms
    train_transform = make_transforms(training=True)
    eval_transform = make_transforms(training=False)
    
    # Init datasets
    train_dataset = DownstreamDataset(train_df, args.mri_folder, mri_transforms=train_transform)
    val_dataset = DownstreamDataset(val_df, args.mri_folder, mri_transforms=eval_transform)
    test_dataset = DownstreamDataset(test_df, args.mri_folder, mri_transforms=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 2. Build Model
    print("Building ViT model...")
    # Lấy param theo config của mri_vit_base_ep300.yaml
    encoder = build_encoder(args)

    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # I-JEPA thường dùng target_encoder tốt hơn để downstream
        state_dict = ckpt.get('target_encoder', ckpt.get('encoder', {})) 
        
        # Xóa tiền tố "module." nếu pretrained được serialize bởi DistributedDataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            k_new = k[7:] if k.startswith("module.") else k
            new_state_dict[k_new] = v
            
        msg = encoder.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {args.ckpt_path}: {msg}")
    else:
        print(f"Warning: checkpoint {args.ckpt_path} not found. Training from scratch.")
        
    model = ViTClassifier(encoder, num_classes=5)
    
    # 3. Setup Strategy
    set_requires_grad(model, args.strategy, args.unfreeze_last_n)
    model.to(device)

    # 4-6. Run the shared training + evaluation protocol
    run_downstream_experiment(model, train_loader, val_loader, test_loader, device, args)


def run_downstream_experiment(model, train_loader, val_loader, test_loader, device, args):
    # 4. Optimizer: layer-wise LR decay + loại weight decay cho norm/bias/pos_embed.
    #    Đây là mấu chốt để full finetuning KHÔNG phá hỏng feature pretrained:
    #    block càng thấp LR càng nhỏ (giữ feature), head + LN cuối LR = peak_lr.
    param_groups = build_param_groups_lrd(
        model, base_lr=args.lr, weight_decay=args.weight_decay, layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    criterion = FocalLoss(gamma=1.2)

    # Warmup tuyến tính -> cosine decay xuống min_lr (step theo mỗi iteration).
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=len(train_loader))

    group_lrs = [g["lr"] for g in param_groups]
    print(f"=> Optimizer AdamW | {len(param_groups)} groups | peak_lr={args.lr:g} "
          f"layer_decay={args.layer_decay} | lr[min={min(group_lrs):.2e}, max={max(group_lrs):.2e}] "
          f"| wd={args.weight_decay}")
    print(f"=> Scheduler: {args.warmup_epochs:g} warmup epoch(s) -> cosine to min_lr={args.min_lr:g} "
          f"| drop_path={0.0 if args.strategy == 'linear_probe' else args.drop_path:g}")

    # 5. Training Loop
    best_val_acc = 0.0
    
    history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [],
    "train_bacc": [], "val_bacc": [],
    "train_qwk": [], "val_qwk": [],
    }
    
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, "best_downstream_model.pth")
    
    print("\nSkip Training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (images, labels) in enumerate(train_loader_tqdm):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            m = nn.Softmax(dim=1)
            loss = criterion(m(outputs), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")
                
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_bacc = balanced_accuracy_score(train_targets, train_preds)
        train_qwk = cohen_kappa_score(train_targets, train_preds, weights="quadratic")

        # Validation Loop
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_loader_tqdm:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                m = nn.Softmax(dim=1)
                loss = criterion(m(outputs), labels)
                
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                val_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")
                
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_bacc = balanced_accuracy_score(val_targets, val_preds)
        val_qwk = cohen_kappa_score(val_targets, val_preds, weights="quadratic")
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train BAcc: {train_bacc:.4f} | Train QWK: {train_qwk:.4f}")
        print(f"Val Loss: {val_loss:.4f}  | Val Acc: {val_acc:.4f} | Val BAcc: {val_bacc:.4f} | Val QWK: {val_qwk:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_bacc"].append(train_bacc)
        history["val_bacc"].append(val_bacc)
        history["train_qwk"].append(train_qwk)
        history["val_qwk"].append(val_qwk)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f">>> Saved new best model with Val Acc {val_acc:.4f}!")
            patience_counter = 0
            # Lưu checkpoint theo epoch (chỉ khi cải thiện và >= save_after_epoch)
            if (epoch + 1) >= args.save_after_epoch:
                ep_ckpt = os.path.join(args.output_dir, f"ckpt_epoch{epoch+1}.pth")
                torch.save(model.state_dict(), ep_ckpt)
                prune_checkpoints(args.output_dir, args.keep_last_n)
                print(f">>> Saved epoch checkpoint: {ep_ckpt}")
        else:
            patience_counter += 1
            print(f">>> No improvement. Patience counter: {patience_counter}/{10}")
            if patience_counter >= 100:
                print("Early stopping triggered.")
                break

    # Plot training history
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs_range, history["train_loss"], label="Train")
    axes[0, 0].plot(epochs_range, history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend()

    axes[0, 1].plot(epochs_range, history["train_acc"], label="Train")
    axes[0, 1].plot(epochs_range, history["val_acc"], label="Val")
    axes[0, 1].set_title("Accuracy"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend()

    axes[1, 0].plot(epochs_range, history["train_bacc"], label="Train")
    axes[1, 0].plot(epochs_range, history["val_bacc"], label="Val")
    axes[1, 0].set_title("Balanced Accuracy"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].legend()

    axes[1, 1].plot(epochs_range, history["train_qwk"], label="Train")
    axes[1, 1].plot(epochs_range, history["val_qwk"], label="Val")
    axes[1, 1].set_title("Quadratic Weighted Kappa"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved training history plot to {plot_path}")

    # Log history ra CSV
    history_csv = os.path.join(args.output_dir, "training_history.csv")
    with open(history_csv, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(history.keys())
        writer.writerow(["epoch"] + keys)
        for i in range(len(history["train_loss"])):
            writer.writerow([i + 1] + [history[k][i] for k in keys])
    print(f"Saved training history CSV to {history_csv}")

    # 6. Final Evaluation
    print("\n===============================")
    print("Final Evaluation...")

    eval_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    # Tập hợp các checkpoint cần eval: best + 5 latest theo epoch
    ckpt_list = []
    if os.path.exists(best_model_path):
        ckpt_list.append(("best", best_model_path))

    epoch_ckpts = glob.glob(os.path.join(args.output_dir, "ckpt_epoch*.pth"))

    def _epoch_of(p):
        m = re.search(r"ckpt_epoch(\d+)\.pth", os.path.basename(p))
        return int(m.group(1)) if m else -1

    epoch_ckpts = sorted(epoch_ckpts, key=_epoch_of)[-args.keep_last_n:]
    for p in epoch_ckpts:
        ckpt_list.append((f"epoch{_epoch_of(p)}", p))

    results = []
    for tag, path in ckpt_list:
        print(f"\n>>> Evaluating checkpoint: {tag} ({path})")
        model.load_state_dict(torch.load(path, map_location=device))
        row = {"checkpoint": tag, "path": path}
        for split, loader in eval_loaders.items():
            metrics = evaluate(model, loader, device, desc=f"[{tag}:{split}]")
            for mk, mv in metrics.items():
                row[f"{split}_{mk}"] = mv
            print(f"    {split}: Acc={metrics['acc']:.4f} | "
                  f"BAcc={metrics['bacc']:.4f} | QWK={metrics['qwk']:.4f}")
        results.append(row)

    # Output kết quả eval ra CSV + JSON
    eval_csv = os.path.join(args.output_dir, "eval_results.csv")
    fieldnames = list(results[0].keys()) if results else []
    with open(eval_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved evaluation results to {eval_csv}")

    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()