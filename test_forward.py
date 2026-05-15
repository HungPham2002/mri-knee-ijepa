import torch
from src.models.vision_transformer import vit_base, vit_predictor
from src.masks.multiblock import MaskCollator

def test():
    print("Initialising models...")
    # Encoder
    encoder = vit_base(patch_size=(12, 16, 16), in_chans=1)
    predictor = vit_predictor(
        num_patches=1000, 
        embed_dim=encoder.embed_dim, 
        predictor_embed_dim=384,
        predictor_grid_size=(10, 10, 10)
    )

    print("Initialising Mask Collator...")
    mask_collator = MaskCollator(
        input_size=(120, 160, 160),
        patch_size=(12, 16, 16),
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10
    )

    print("Creating dummy batch...")

    dummy_batch = [(torch.randn(1, 120, 160, 160), 0) for _ in range(4)]
    
    print("Generating Masks...")
    collated_batch, collated_masks_enc, collated_masks_pred = mask_collator(dummy_batch)
    imgs = collated_batch[0]
    print(f"Images shape: {imgs.shape}")
    print(f"Encoder masks: {len(collated_masks_enc)} masks, Example length: {collated_masks_enc[0].shape}")
    print(f"Predictor masks: {len(collated_masks_pred)} masks, Example length: {collated_masks_pred[0].shape}")

    print("Forward Pass: Encoder...")
    z = encoder(imgs, collated_masks_enc)
    print(f"Encoder output shape: {z.shape}")

    print("Forward Pass: Predictor...")
    # I-JEPA predictor signature: forward(self, x, masks_x, masks)
    preds = predictor(z, collated_masks_enc, collated_masks_pred)
    print(f"Predictor output shape: {preds.shape}")
    
    print("Test passed successfully!")

if __name__ == '__main__':
    test()
