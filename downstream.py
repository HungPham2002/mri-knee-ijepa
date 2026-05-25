import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from focal_loss.focal_loss import FocalLoss
from sklearn.metrics import accuracy_score, balanced_accuracy_score

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
            # Fallback nếu file lỗi
            mri_data = np.zeros((120, 160, 160), dtype=np.float32)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/network-volume/hungph-data/data/SAG_3D_DESS_v2_full")
    parser.add_argument("--mri_folder", type=str, default="/network-volume/hungph-data/data/SAG_3D_DESS_v2_full/MRI_Numpy")
    parser.add_argument("--ckpt_path", type=str, default="/network-volume/hungph-data/mri-knee-ijepa/logs/mri_vit_base/mri_vit_base_300ep-ep100.pth.tar")
    parser.add_argument("--strategy", type=str, choices=["linear_probe", "partial", "full"], default="linear_probe",
                        help="Tùy chọn fine-tune.")
    parser.add_argument("--unfreeze_last_n", type=int, default=4, help="Số block cuối của ViT cần unfreeze (nếu strategy=partial).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="downstream_output")
    
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
    encoder = vit.vit_base(img_size=[120, 160, 160], patch_size=(12, 16, 16))
    
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
    
    # 4. Setup Optimizer
    # QUAN TRỌNG: Chỉ set parameters cần update gradient cho AdamW
    # Loại bỏ weight decay cho các bias/LayerNorm 
    params_to_optimize = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim <= 1 or 'bias' in name or 'norm' in name:
                # Không áp dụng weight decay
                params_to_optimize.append({'params': [param], 'weight_decay': 0.0})
            else:
                params_to_optimize.append({'params': [param], 'weight_decay': args.weight_decay})
                
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr)
    criterion = FocalLoss(gamma=1.2)
    
    # 5. Training Loop
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_downstream_model.pth")
    
    print("\nStarting Training...")
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
            
            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}")
                
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_bacc = balanced_accuracy_score(train_targets, train_preds)
        
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
        
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train BAcc: {train_bacc:.4f}")
        print(f"Val Loss: {val_loss:.4f}  | Val Acc: {val_acc:.4f} | Val BAcc: {val_bacc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f">>> Saved new best model with Val Acc {val_acc:.4f}!")
            
    # 6. Test Loop
    print("\n===============================")
    print("Evaluating on Test Set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        
    model.eval()
    test_preds, test_targets = [], []
    
    test_loader_tqdm = tqdm(test_loader, desc="[Test]")
    with torch.no_grad():
        for images, labels in test_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    test_bacc = balanced_accuracy_score(test_targets, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Balanced Accuracy: {test_bacc:.4f}")

if __name__ == "__main__":
    main()
