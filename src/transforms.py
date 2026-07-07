"""
Pipeline transform 3D tự viết cho MRI (R1) — thay thế torchio.

Thiết kế (xem contexts/R1_context.md §4.1):
  - Thao tác trên np.ndarray shape (1, D, H, W) float32, trả về torch.FloatTensor (1, D, H, W).
  - Chỉ phụ thuộc numpy + torch (skimage optional cho fg_method="otsu").
  - Mọi transform là **class callable** (picklable) => an toàn với num_workers>0
    (KHÔNG dùng lambda/closure có state).
  - training=True  -> Compose([ForegroundZScore, RandomResizedCrop3D, RandomFlip3D, (RandomAffine3D)])
    training=False -> Compose([ForegroundZScore])  # xác định, không random

Lý do MRI-specific:
  - ForegroundZScore thay min-max: MRI không calibrate như CT; z-score trên mô (bỏ nền)
    cho input variance ổn định giữa bệnh nhân, giảm một nguyên nhân collapse.
  - RandomResizedCrop3D là nguồn đa dạng không gian cốt lõi của I-JEPA (bản cũ thiếu hoàn toàn).
  - Cắt bỏ augmentation nặng (motion/noise/blur/gamma/bias) vì bơm nhiễu vào target EMA.
"""

from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F

logger = getLogger()


class ForegroundZScore(object):
    """Chuẩn hóa z-score trên foreground (áp dụng cả train lẫn eval => giống hệt nhau).

    - Ước lượng foreground mask theo `fg_method`:
        * "percentile" (mặc định): thr = percentile(x, p_bg); fg = x > thr
        * "nonzero": fg = x > eps
        * "otsu": dùng skimage.filters.threshold_otsu; nếu import lỗi -> fallback percentile (warn).
    - mu, sigma = mean/std trên foreground; nếu fg rỗng hoặc sigma ~ 0 -> fallback toàn volume.
    - x = (x - mu) / (sigma + 1e-6); clip [-clip, clip].
    """

    def __init__(self, fg_method="percentile", p_bg=20, eps=1e-6, clip=5.0):
        self.fg_method = fg_method
        self.p_bg = p_bg
        self.eps = eps
        self.clip = clip

    def _foreground_mask(self, x):
        if self.fg_method == "nonzero":
            return x > self.eps
        if self.fg_method == "otsu":
            try:
                from skimage.filters import threshold_otsu
                thr = threshold_otsu(x)
                return x > thr
            except Exception:
                logger.warning("ForegroundZScore: otsu không khả dụng (skimage?), fallback percentile.")
        # percentile (mặc định + fallback)
        thr = np.percentile(x, self.p_bg)
        return x > thr

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        fg = self._foreground_mask(x)
        vals = x[fg]
        if vals.size == 0:
            vals = x.reshape(-1)  # fallback: toàn volume
        mu = float(vals.mean())
        sigma = float(vals.std())
        if sigma < self.eps:
            # foreground gần như hằng số -> dùng thống kê toàn volume
            allv = x.reshape(-1)
            mu, sigma = float(allv.mean()), float(allv.std())
        x = (x - mu) / (sigma + self.eps)
        x = np.clip(x, -self.clip, self.clip)
        return x.astype(np.float32)


class MinMaxNorm(object):
    """Min-max per-volume về [out_min, out_max] (tái tạo R0: tio.RescaleIntensity(-1,1)).

    Chỉ dùng khi normalize="minmax" để ablate/so R0. Nhạy outlier (một voxel sáng nén
    hết tương phản) -> KHÔNG khuyến nghị cho R1, chỉ để reproduce.
    """

    def __init__(self, out_min=-1.0, out_max=1.0, eps=1e-8):
        self.out_min = out_min
        self.out_max = out_max
        self.eps = eps

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        lo, hi = float(x.min()), float(x.max())
        x = (x - lo) / (hi - lo + self.eps)
        x = x * (self.out_max - self.out_min) + self.out_min
        return x.astype(np.float32)


class RandomResizedCrop3D(object):
    """Multi-scale crop 3D thật (CHỈ training).

    - Sample scale s ~ U[crop_scale_min, crop_scale_max] (mặc định [0.5, 1.0]).
      KHÔNG dùng cận dưới quá nhỏ (vd 0.2): ROI đã crop chặt quanh khớp, scale quá nhỏ
      crop *mất* khe khớp -> hỏng tín hiệu KL grading. (MRI-specific)
    - Kích thước crop theo cùng tỉ lệ tuyến tính mỗi trục: d=round(s*D), h=round(s*H),
      w=round(s*W) (giữ tỉ lệ khung ROI, không aspect distortion).
    - Vị trí crop random nhưng đảm bảo đủ foreground: nếu fraction fg trong crop < tau
      thì resample tối đa K lần rồi lấy crop trung tâm.
    - Resize crop về crop_size bằng F.interpolate(mode='trilinear', align_corners=False).

    Random state: dùng RNG cục bộ nếu `seed` được set (eval xác định); mặc định dùng
    torch RNG toàn cục (đã seed ở train.py). Không dùng closure có state (picklable).
    """

    def __init__(self, crop_size=(120, 160, 160), crop_scale=(0.5, 1.0),
                 fg_frac_tau=0.1, max_tries=10, fg_thr_method="percentile",
                 p_bg=20, seed=None):
        self.crop_size = tuple(int(c) for c in crop_size)
        self.crop_scale = (float(crop_scale[0]), float(crop_scale[1]))
        self.fg_frac_tau = fg_frac_tau
        self.max_tries = max_tries
        self.fg_thr_method = fg_thr_method
        self.p_bg = p_bg
        self.seed = seed

    def _rng(self):
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(int(self.seed))
            return g
        return None  # dùng RNG toàn cục

    def _rand(self, g):
        return torch.rand(1, generator=g).item()

    def _randint(self, high, g):
        # trả về int trong [0, high) ; high>=1
        if high <= 1:
            return 0
        return int(torch.randint(0, high, (1,), generator=g).item())

    def _fg_thr(self, x):
        # ngưỡng foreground để đo fraction (nhất quán với ForegroundZScore percentile)
        if self.fg_thr_method == "nonzero":
            return 1e-6
        return np.percentile(x, self.p_bg)

    def __call__(self, x):
        # x: (1, D, H, W) float32 (đã z-score)
        x = np.asarray(x, dtype=np.float32)
        _, D, H, W = x.shape
        g = self._rng()

        s = self.crop_scale[0] + self._rand(g) * (self.crop_scale[1] - self.crop_scale[0])
        d = max(1, min(D, int(round(s * D))))
        h = max(1, min(H, int(round(s * H))))
        w = max(1, min(W, int(round(s * W))))

        thr = self._fg_thr(x)
        vol_crop = float(d * h * w)

        best = None
        for _ in range(self.max_tries):
            top_d = self._randint(D - d + 1, g)
            top_h = self._randint(H - h + 1, g)
            left_w = self._randint(W - w + 1, g)
            crop = x[:, top_d:top_d + d, top_h:top_h + h, left_w:left_w + w]
            fg_frac = float((crop > thr).sum()) / vol_crop
            if best is None or fg_frac > best[1]:
                best = ((top_d, top_h, left_w), fg_frac)
            if fg_frac >= self.fg_frac_tau:
                break
        else:
            # hết tries mà chưa đủ foreground -> crop trung tâm
            top_d = max(0, (D - d) // 2)
            top_h = max(0, (H - h) // 2)
            left_w = max(0, (W - w) // 2)
            best = ((top_d, top_h, left_w), best[1] if best else 0.0)

        (top_d, top_h, left_w), _ = best
        crop = x[:, top_d:top_d + d, top_h:top_h + h, left_w:left_w + w]

        # Resize về crop_size bằng trilinear (thêm batch dim)
        t = torch.from_numpy(np.ascontiguousarray(crop)).unsqueeze(0)  # (1,1,d,h,w)
        t = F.interpolate(t, size=self.crop_size, mode="trilinear", align_corners=False)
        return t.squeeze(0).numpy().astype(np.float32)  # (1, *crop_size)


class RandomFlip3D(object):
    """Lật theo MỘT trục cấu hình (mặc định trục cuối = width), p=0.5. CHỈ training.

    KHÔNG lật bừa mọi trục (một số trục giải phẫu lật lên là phi vật lý). flip_axis
    tính trên tensor (1, D, H, W): -1 (width), -2 (height), -3 (depth). Trục để trong
    config để dễ chỉnh khi biết orientation thật.
    """

    def __init__(self, flip_axis=-1, p=0.5, seed=None):
        self.flip_axis = flip_axis
        self.p = p
        self.seed = seed

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(int(self.seed))
            r = torch.rand(1, generator=g).item()
        else:
            r = torch.rand(1).item()
        if r < self.p:
            x = np.flip(x, axis=self.flip_axis)
        return np.ascontiguousarray(x, dtype=np.float32)


class RandomAffine3D(object):
    """(Tùy chọn, mặc định TẮT) affine nhẹ: rotation nhỏ + scale nhỏ qua grid_sample 3D.

    Bật bằng cờ use_affine. Giữ nhẹ để không phá tín hiệu giải phẫu.
    """

    def __init__(self, max_rot_deg=8.0, max_scale=0.05, p=0.5, seed=None):
        self.max_rot = max_rot_deg
        self.max_scale = max_scale
        self.p = p
        self.seed = seed

    def _g(self):
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(int(self.seed))
            return g
        return None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        g = self._g()
        if torch.rand(1, generator=g).item() >= self.p:
            return x
        t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)  # (1,1,D,H,W)

        # rotation quanh trục sâu (plane H-W) nhẹ + scale đẳng hướng
        deg = (torch.rand(1, generator=g).item() * 2 - 1) * self.max_rot
        theta_rad = np.deg2rad(deg)
        sc = 1.0 + (torch.rand(1, generator=g).item() * 2 - 1) * self.max_scale
        cos, sin = np.cos(theta_rad) / sc, np.sin(theta_rad) / sc

        # affine 3D theta (2x4 -> ở đây 3x4). Xoay trong mặt phẳng (h,w), depth giữ nguyên/scale.
        theta = torch.tensor([
            [1.0 / sc, 0.0, 0.0, 0.0],
            [0.0, cos, -sin, 0.0],
            [0.0, sin, cos, 0.0],
        ], dtype=torch.float32).unsqueeze(0)

        grid = F.affine_grid(theta, t.shape, align_corners=False)
        t = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return t.squeeze(0).numpy().astype(np.float32)


class Compose(object):
    """Compose picklable (không dùng torchvision để tránh phụ thuộc thừa)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class ToTensor3D(object):
    """np.ndarray (1, D, H, W) -> torch.FloatTensor (1, D, H, W)."""

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def make_transforms(training=True,
                    crop_size=(120, 160, 160),
                    crop_scale=(0.5, 1.0),
                    normalize="foreground_zscore",
                    fg_method="percentile",
                    p_bg=20,
                    flip_axis=-1,
                    use_flip=True,
                    use_affine=False,
                    **kwargs):
    """Xây pipeline transform 3D cho MRI.

    Chữ ký giữ tương thích call-site cũ; vẫn nhận **kwargs để nuốt tham số cũ
    (color_jitter, gaussian_blur, horizontal_flip, ...) mà KHÔNG lỗi, nhưng bỏ qua chúng.

    normalize:
      - "foreground_zscore" (R1, mặc định): ForegroundZScore.
      - "minmax" (R0): MinMaxNorm(-1,1) để ablate/reproduce.

    training=True  -> norm + RandomResizedCrop3D + (RandomFlip3D) + (RandomAffine3D)
    training=False -> norm  (xác định; train/eval nhận normalization giống hệt nhau)
    """
    if normalize == "minmax":
        norm = MinMaxNorm(out_min=-1.0, out_max=1.0)
    else:
        norm = ForegroundZScore(fg_method=fg_method, p_bg=p_bg)

    if not training:
        return Compose([norm, ToTensor3D()])

    steps = [
        norm,
        RandomResizedCrop3D(crop_size=crop_size, crop_scale=crop_scale,
                            fg_thr_method=fg_method, p_bg=p_bg),
    ]
    if use_flip:
        steps.append(RandomFlip3D(flip_axis=flip_axis))
    if use_affine:
        steps.append(RandomAffine3D())
    steps.append(ToTensor3D())
    return Compose(steps)
