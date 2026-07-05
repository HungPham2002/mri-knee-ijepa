#!/usr/bin/env bash
# Fair benchmark: backbone khởi tạo từ ImageNet-21k (inflated 2D->3D).
# Cùng protocol với downstream.py (I-JEPA). Lần chạy đầu sẽ download + cache
# trọng số timm vào --weights_cache; các lần sau nạp lại từ cache.
set -e

python downstream_in21k.py \
--data_root /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full \
--mri_folder /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full/MRI_Numpy \
--strategy linear_probe \
--unfreeze_last_n 4 \
--output_dir /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/downstream_in21k_linear_probe

python downstream_in21k.py \
--data_root /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full \
--mri_folder /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full/MRI_Numpy \
--strategy partial \
--unfreeze_last_n 4 \
--output_dir /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/downstream_in21k_partial

python downstream_in21k.py \
--data_root /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full \
--mri_folder /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full/MRI_Numpy \
--strategy full \
--output_dir /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/downstream_in21k_full
