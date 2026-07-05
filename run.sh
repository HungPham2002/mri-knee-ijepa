python downstream.py \
--data_root /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full \
--mri_folder /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full/MRI_Numpy \
--ckpt_path /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/mri_vit_base/mri_vit_base_300ep-ep200.pth.tar \
--strategy linear_probe \
--unfreeze_last_n 4 \
--output_dir /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/downstream_ep200_linear_probe \

python downstream.py \
--data_root /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full \
--mri_folder /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full/MRI_Numpy \
--ckpt_path /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/mri_vit_base/mri_vit_base_300ep-ep200.pth.tar \
--strategy partial \
--unfreeze_last_n 4 \
--output_dir /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/downstream_ep200_partial \

python downstream.py \
--data_root /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full \
--mri_folder /home/ubuntu/ecd_hungpham/data/SAG_3D_DESS_v2_full/MRI_Numpy \
--ckpt_path /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/mri_vit_base/mri_vit_base_300ep-ep200.pth.tar \
--strategy full \
--output_dir /home/ubuntu/ecd_hungpham/mri-knee-ijepa/logs/downstream_ep200_full \