# MRI-Knee I-JEPA — Chuẩn hóa Vanilla (R1)

Pretraining tự giám sát (self-supervised) cho **MRI khớp gối 3D** bằng **I-JEPA**,
downstream là **KL grading** (phân loại ordinal 5 lớp mức độ thoái hóa khớp).

- Input volume: `(1, 120, 160, 160)` — patch `(12, 16, 16)` → grid `10×10×10 = 1000` patch.
- Backbone: `vit_base` (ViT-B) 3D.
- Đây là bản **adapt I-JEPA từ 2D → 3D**. Tài liệu này mô tả **R1**: chuẩn hóa lại bản
  vanilla để nó chạy *đúng như thiết kế*, *phù hợp ảnh MRI*, và *ổn định (không collapse)*.

---

## 1. Vấn đề (bản vanilla R0 bị gì?)

Bản I-JEPA 3D ban đầu (**R0**) chạy được nhưng chất lượng biểu diễn *chưa tối đa*:

- **Representation collapse sớm.** Pretraining loss rơi về ~0.0015 ngay epoch 1, giữ thấp
  tới ~ep100, rồi *tăng* và plateau ~0.05 khi biểu diễn de-collapse. Hệ quả trớ trêu:
  checkpoint loss-thấp (ep100) cho feature **kém hơn** checkpoint loss-cao (ep300).
- **Benchmark công bằng (đã chạy):** test QWK — partial FT: I-JEPA **0.729** vs IN21k 0.728
  (hòa); linear-probe: **0.513** vs 0.586 (thua rõ → feature chưa đủ tách được lớp).
  → pretraining *có tác dụng lớn*, nhưng representation *chưa mạnh nhất có thể*.

Nguyên nhân gốc là **5 nhóm lỗi** khiến vanilla vừa chạy **sai thiết kế** vừa **dễ collapse**.

---

## 2. R1 đã sửa gì (5 lỗi + cách fix)

| # | Lỗi của R0 | Cách R1 fix | Cờ ablate (đặt = R0) |
|---|---|---|---|
| 1 | `make_transforms` **bỏ qua config** → KHÔNG có multi-scale crop (mất nguồn đa dạng không gian cốt lõi của I-JEPA) | Viết `RandomResizedCrop3D` thật (crop scale rồi resize trilinear về `crop_size`); `make_transforms` **đọc thật** `crop_size/crop_scale` | `crop_scale: [0.2,1.0]` (cũ) |
| 2 | Normalization **min-max** nhạy outlier → input variance thấp → đường vào collapse | `ForegroundZScore`: z-score trên mô (bỏ nền), clip `[-5,5]`; train/eval **giống hệt nhau** | `normalize: minmax` |
| 3 | Augment nặng (motion/noise/blur/gamma/bias) **bơm nhiễu vào target EMA** | Bỏ hết augment *cường độ*; chỉ giữ biến đổi *hình học* (RRC + flip) | — (thiết kế R1) |
| 4 | Masking **erosion**: clamp `while >=` ép enc-block xuống ≤ grid−1 → context thực chỉ **~24.5%** (config ghi 85%) | Đổi sang `min(d, grid)`; thuật toán masking **giữ nguyên**; thêm `log_effective_fractions()` | `fix_enc_clamp: false` |
| 5 | Hyperparam chưa scale theo batch → LR quá cao, warmup quá dài, EMA base thấp → collapse | Config mới: batch ↑, lr ↓, warmup ↓, ema base ↑, final_wd ↓ | đặt lại lr/warmup/ema cũ |

**Kết quả smoke-test (đã chạy trên data thật):**
- Masking: **context R1 = 0.421** (mục tiêu 0.4–0.55) vs **R0 = 0.245**.
- Pretrain vài iter: loss **giảm dần** (không sập ~0), `feat_std` **ổn định** (không → 0).
- Không còn phụ thuộc `torchio` trong nhánh pretrain/downstream.

Ngoài ra R1 còn dọn code fragile + thêm instrumentation chống collapse:
- **Predictor `grid_size` lấy từ encoder** (bỏ hardcode `(10,10,10)` — đổi crop_size không còn lệch pos-embed im lặng); `ipe_scale` đặt **tường minh** trong config.
- **Proxy `feat_std`** log trong training loop + cột CSV (collapse → `feat_std → 0`).
- **`eval_effective_rank.py`** — đo **RankMe** (effective rank) theo epoch: trục "chứng minh cơ chế" (R0 rank sập sớm; R1 kỳ vọng giữ cao).
- **Toggle `var_reg_weight`** (VICReg-style, **TẮT sẵn**) — chỉ bật nếu vẫn collapse. *Đây là stabilization mượn, KHÔNG phải đóng góp của paper.*

---

## 3. Pipeline transform — pretraining vs downstream

Augment phục vụ **2 mục đích khác nhau**:

| | Pretraining (`training=True`) | Downstream (mặc định) |
|---|---|---|
| Mục tiêu | Học biểu diễn (SSL) | Đo/so chất lượng biểu diễn (có nhãn) |
| Normalize | ForegroundZScore | ForegroundZScore (giống hệt) |
| RandomResizedCrop3D | **Có** (đa dạng không gian = tín hiệu học) | **Không** |
| Flip 3D | **Có** | **Không** |
| Lý do | crop tạo nhiều "view" → feature ổn định qua scale/vị trí | giữ benchmark 3 backbone **sạch**, tránh crop cắt mất khe khớp (tín hiệu KL) |

> Downstream cố tình dùng transform **normalization-only** cho cả train lẫn eval để
> mọi backbone (I-JEPA R1 / IN21k / from-scratch) nhận **cùng** một transform → chênh
> lệch QWK quy được về *nguồn trọng số*, không phải augment. Muốn bật augment downstream:
> đổi `training=False` → `training=True` ở [`downstream.py`](downstream.py) (và giữ đồng nhất cho cả 3 backbone).

---

## 4. Bản đồ file

| File | Vai trò | R1 |
|---|---|---|
| [`src/transforms.py`](src/transforms.py) | Pipeline transform 3D tự viết (ForegroundZScore, RandomResizedCrop3D, RandomFlip3D, MinMaxNorm) | **Viết lại**, bỏ torchio |
| [`src/masks/multiblock.py`](src/masks/multiblock.py) | `MaskCollator` | Fix erosion (`fix_enc_clamp`) + `log_effective_fractions()` |
| [`src/helper.py`](src/helper.py) | `init_model`, `init_opt` | Truyền `predictor_grid_size` từ encoder |
| [`src/models/vision_transformer.py`](src/models/vision_transformer.py) | ViT + predictor | Predictor nhận grid từ encoder (bỏ hardcode) |
| [`src/train.py`](src/train.py) | Training loop | Call-site mới, đọc cờ config, log `feat_std`, var_reg tùy chọn |
| [`src/datasets/dess_dataset.py`](src/datasets/dess_dataset.py) | `DESSDataset3D` | Tránh double-wrap tensor |
| [`configs/mri_vit_base_R1.yaml`](configs/mri_vit_base_R1.yaml) | **Config R1 (mới)** | — |
| [`configs/mri_vit_base_ep300.yaml`](configs/mri_vit_base_ep300.yaml) | Config R0 | **Giữ nguyên** |
| [`eval_effective_rank.py`](eval_effective_rank.py) | Metric collapse (RankMe) | **Mới** |
| [`downstream.py`](downstream.py), [`downstream_in21k.py`](downstream_in21k.py) | Eval KL grading | Normalization mới (parity) |
| [`run_R1.sh`](run_R1.sh) | Script chạy toàn bộ R1 | **Mới** |

---

## 5. Config R1 — các cờ chính

Mọi thay đổi hành vi gate sau **một cờ**, mặc định = giá trị R1; đặt về giá trị R0 thì
**tái tạo được R0** (để sau còn *ablate* biết gain đến từ đâu). Xem
[`configs/mri_vit_base_R1.yaml`](configs/mri_vit_base_R1.yaml):

```yaml
data:
  crop_scale: [0.5, 1.0]          # multi-scale crop thật; cận dưới 0.5 (MRI-specific)
  normalize: foreground_zscore    # R0='minmax'
  fg_method: percentile           # 'nonzero' | 'otsu'(fallback percentile nếu thiếu skimage)
  p_bg: 20
  use_flip: true
  flip_axis: -1
  use_affine: false               # affine nhẹ, mặc định TẮT
mask:
  fix_enc_clamp: true             # fix erosion (§4.3); false -> R0
optimization:
  ipe_scale: 1.0                  # đặt tường minh (tránh nhầm default 1.25)
  ema: [0.999, 1.0]               # base cao hơn R0 [0.996,1.0] -> target ổn định
  var_reg_weight: 0.0             # VICReg-style stabilization, TẮT sẵn
  # lr / warmup / final_weight_decay / batch_size: hạ/điều chỉnh so R0 (xem file)
```

Để **tái tạo R0** cho ablation: `normalize: minmax`, `fix_enc_clamp: false`,
`ema: [0.996, 1.0]`, `crop_scale: [0.2, 1.0]`, lr/warmup cũ.

---

## 6. Cách chạy

Toàn bộ quy trình gói trong
[`run_R1.sh`](run_R1.sh); hoặc chạy từng bước:

```bash
cd /network-volume/hungph/mri-knee-ijepa
PY=/root/miniconda3/envs/knee/bin/python

# 1) Pretrain I-JEPA R1
$PY main.py --fname configs/mri_vit_base_R1.yaml --devices cuda:0

# 2) Rank-vs-epoch (cơ chế chống collapse)
$PY eval_effective_rank.py \
  --ckpt "logs/mri_vit_base_R1/mri_vit_base_R1-ep*.pth.tar" \
  --data_root /network-volume/hungph/data/SAG_3D_DESS_v2_full \
  --n_samples 256 --out logs/mri_vit_base_R1/rank_vs_epoch.csv

# 3) Downstream KL grading (3 strategy)
CKPT=logs/mri_vit_base_R1/mri_vit_base_R1-ep300.pth.tar
for S in linear_probe partial full; do
  $PY downstream.py --ckpt_path $CKPT --strategy $S --unfreeze_last_n 4 \
    --output_dir logs/downstream_R1_ep300_$S; done

# 4) BASELINE PARITY - chạy lại IN21k + from-scratch với normalization MỚI (z-score).
for S in linear_probe partial full; do
  $PY downstream_in21k.py \
    --data_root "$DATA" --mri_folder "$DATA/MRI_Numpy" \
    --strategy $S --unfreeze_last_n 4 \
    --output_dir "logs/downstream_in21k_R1norm_${S}"
done
$PY downstream_in21k.py \
  --data_root "$DATA" --mri_folder "$DATA/MRI_Numpy" \
  --from_scratch --output_dir "logs/downstream_scratch_R1norm_full"
```

Nếu vẫn collapse: thử **lần lượt** (đừng đổi nhiều thứ cùng lúc) — tăng `ema` base
(0.999→0.9995) → bật `var_reg_weight: 1.0` → kiểm lại `p_bg`/normalization.

---
