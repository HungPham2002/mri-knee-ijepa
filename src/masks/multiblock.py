# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):


    def __init__(
        self,
        input_size=(120, 160, 160),
        patch_size=(10, 16, 16),
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        fix_enc_clamp=False
    ):
        super(MaskCollator, self).__init__()
        if isinstance(input_size, int):
            input_size = (input_size, ) * 3
        if isinstance(patch_size, int):
            patch_size = (patch_size, ) * 3
        self.patch_size = patch_size
        self.depth = input_size[0] // patch_size[0]
        self.height, self.width = input_size[1] // patch_size[1], input_size[2] // patch_size[2]
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        # R1 §4.3: khi True, cho phép cạnh block đạt tối đa = grid dim (dùng min thay vì
        # clamp `while >=` vốn ép cạnh xuống <= grid-1). Sửa erosion không mong muốn khiến
        # enc block không bao giờ đạt scale cấu hình (grid chỉ 10). False -> hành vi R0.
        self.fix_enc_clamp = fix_enc_clamp
        self._itr_counter = Value('i', -1)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.depth * self.height * self.width * mask_scale)
        
        min_ar, max_ar = aspect_ratio_scale
        _rands = torch.rand(2, generator=generator).tolist()
        ar_d_h = min_ar + _rands[0] * (max_ar - min_ar)
        ar_h_w = min_ar + _rands[1] * (max_ar - min_ar)
        
        h = int(round((max_keep * ar_h_w / ar_d_h) ** (1 / 3.0)))
        w = int(round(h / ar_h_w))
        d = int(round(h * ar_d_h))
        
        if self.fix_enc_clamp:
            # R1: cho phép cạnh đạt tối đa = grid dim (không kéo xuống dưới scale cấu hình).
            # Pred block (scale nhỏ) cạnh nhỏ nên không chạm clamp -> chỉ enc block hưởng lợi.
            d = min(d, self.depth)
            h = min(h, self.height)
            w = min(w, self.width)
        else:
            # R0: erosion cũ ép cạnh xuống <= grid-1.
            while d >= self.depth: d -= 1
            while h >= self.height: h -= 1
            while w >= self.width: w -= 1

        d = max(1, d)
        h = max(1, h)
        w = max(1, w)

        return (d, h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        d, h, w = b_size

        def constrain_mask(mask, tries=0):
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            top_d = torch.randint(0, self.depth - d + 1, (1,)).item() if self.depth > d else 0
            top_h = torch.randint(0, self.height - h + 1, (1,)).item() if self.height > h else 0
            left_w = torch.randint(0, self.width - w + 1, (1,)).item() if self.width > w else 0
            
            mask = torch.zeros((self.depth, self.height, self.width), dtype=torch.int32)
            mask[top_d:top_d+d, top_h:top_h+h, left_w:left_w+w] = 1
            
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask_nz = torch.nonzero(mask.flatten())
            valid_mask = len(mask_nz) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask_nz.squeeze()
        mask_complement = torch.ones((self.depth, self.height, self.width), dtype=torch.int32)
        mask_complement[top_d:top_d+d, top_h:top_h+h, left_w:left_w+w] = 0
        
        return mask, mask_complement

    def log_effective_fractions(self, batch_size=8, num_batches=4):
        """Đo minh bạch mask fractions bằng cách chạy collator trên vài batch giả.

        Trả về dict:
          - context_fraction: mean(len(enc_mask)) / num_patches  (context thực encoder thấy)
          - target_fraction_per_block: mean(len(pred_mask)) / num_patches
          - total_target_fraction: target_fraction_per_block * npred (xấp xỉ, có thể overlap)
        Không ảnh hưởng training; chỉ để log INFO lúc khởi động.
        """
        num_patches = self.depth * self.height * self.width
        enc_lens, pred_lens = [], []
        dummy = [(torch.zeros(1, *[self.patch_size[i] * s for i, s in
                                   enumerate((self.depth, self.height, self.width))]), 0)
                 for _ in range(batch_size)]
        for _ in range(num_batches):
            _, masks_enc, masks_pred = self.__call__(dummy)
            # masks_enc: list(nenc) of tensor (B, keep_enc); masks_pred: list(npred) of (B, keep_pred)
            enc_lens.append(float(masks_enc[0].shape[1]))
            pred_lens.append(float(masks_pred[0].shape[1]))
        ctx = sum(enc_lens) / len(enc_lens) / num_patches
        tgt = sum(pred_lens) / len(pred_lens) / num_patches
        return {
            'num_patches': num_patches,
            'context_fraction': ctx,
            'target_fraction_per_block': tgt,
            'total_target_fraction': tgt * self.npred,
        }

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.depth * self.height * self.width
        min_keep_enc = self.depth * self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
