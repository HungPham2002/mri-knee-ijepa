#!/usr/bin/env bash
# Fair benchmark: backbone khởi tạo từ ImageNet-21k (inflated 2D->3D).
# Cùng protocol với downstream.py (I-JEPA). Lần chạy đầu sẽ download + cache
# trọng số timm vào --weights_cache; các lần sau nạp lại từ cache.
set -e

python downstream_in21k.py \
--data_root /network-volume/hungph/data/SAG_3D_DESS_v2_full \
--mri_folder /network-volume/hungph/data/SAG_3D_DESS_v2_full/MRI_Numpy \
--strategy linear_probe \
--unfreeze_last_n 4 \
--output_dir /network-volume/hungph/mri-knee-ijepa/logs/downstream_in21k_linear_probe

python downstream_in21k.py \
--data_root /network-volume/hungph/data/SAG_3D_DESS_v2_full \
--mri_folder /network-volume/hungph/data/SAG_3D_DESS_v2_full/MRI_Numpy \
--strategy partial \
--unfreeze_last_n 4 \
--output_dir /network-volume/hungph/mri-knee-ijepa/logs/downstream_in21k_partial
