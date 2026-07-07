#!/usr/bin/env bash
# R1 - chuẩn hóa vanilla I-JEPA 3D cho MRI.
PY=/root/miniconda3/envs/knee/bin/python
REPO=/network-volume/hungph/mri-knee-ijepa
DATA=/network-volume/hungph/data/SAG_3D_DESS_v2_full
cd "$REPO"


# =========================================================================
# 1) TRAINING (I-JEPA R1) - 300 epochs
# =========================================================================
$PY main.py --fname configs/mri_vit_base_R1.yaml --devices cuda:0

# =========================================================================
# 2) RANK-vs-EPOCH (cơ chế chống collapse) - sau khi có checkpoint.
# =========================================================================
$PY eval_effective_rank.py \
  --ckpt "logs/mri_vit_base_R1/mri_vit_base_R1-ep*.pth.tar" \
  --data_root "$DATA" --n_samples 256 --out logs/mri_vit_base_R1/rank_vs_epoch.csv

# =========================================================================
# 3) DOWNSTREAM (I-JEPA R1) - 3 strategy. Đổi --ckpt_path sang epoch muốn eval.
# =========================================================================
CKPT=logs/mri_vit_base_R1/mri_vit_base_R1-ep300.pth.tar
for S in linear_probe partial full; do
  $PY downstream.py \
    --data_root "$DATA" --mri_folder "$DATA/MRI_Numpy" \
    --ckpt_path "$CKPT" --strategy $S --unfreeze_last_n 4 \
    --output_dir "logs/downstream_R1_ep300_${S}"
done

# =========================================================================
# 4) BASELINE PARITY - chạy lại IN21k + from-scratch với normalization MỚI (z-score).
#    (bắt buộc để so sánh công bằng)
# =========================================================================
for S in linear_probe partial full; do
  $PY downstream_in21k.py \
    --data_root "$DATA" --mri_folder "$DATA/MRI_Numpy" \
    --strategy $S --unfreeze_last_n 4 \
    --output_dir "logs/downstream_in21k_R1norm_${S}"
done
$PY downstream_in21k.py \
  --data_root "$DATA" --mri_folder "$DATA/MRI_Numpy" \
  --from_scratch --output_dir "logs/downstream_scratch_R1norm_full"
