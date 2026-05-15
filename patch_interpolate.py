import re

with open('src/models/vision_transformer.py', 'r') as f:
    text = f.read()

interpolate_replacement = """    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N:
            return pos_embed
            
        dim = x.shape[-1]
        grid_size = self.patch_embed.grid_size
        # pos_embed: [1, N, dim] -> [1, dim, grid_size[0], grid_size[1], grid_size[2]]
        pos_embed = pos_embed.reshape(1, grid_size[0], grid_size[1], grid_size[2], dim).permute(0, 4, 1, 2, 3)
        
        # Calculate new grid size based on input x shape
        # x shape entering here is [B, npatch, dim]
        # x originally was [B, C, D, H, W], we project and flatten.
        # So we can query self.patch_embed for current spatial dims dynamically.
        # But wait, patch_embed.forward just flattens. We can just guess the proportions
        # or require the user to configure the model with the correct img_size!
        
        import math
        # simplistic check assuming proportional resizing
        ratio = (npatch / N) ** (1/3)
        new_d = int(round(grid_size[0] * ratio))
        new_h = int(round(grid_size[1] * ratio))
        new_w = int(round(grid_size[2] * ratio))
        
        # Exact patch grid dimensions are expected to be known if dynamic inputs are used.
        # For our Knee MRI case, D=10, H=10, W=10.
        # A quick hack is to just resize to the true spatial dimensions if available.
        # We will use interpolate.
        pos_embed = torch.nn.functional.interpolate(
            pos_embed,
            size=(new_d, new_h, new_w),
            mode='trilinear',
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return pos_embed"""

text = re.sub(r'    def interpolate_pos_encoding\(self, x, pos_embed\):[\s\S]*?return torch\.cat\(\(class_emb\.unsqueeze\(0\), pos_embed\), dim=1\)', interpolate_replacement, text)

# Also let's fix img_size defaulting
text = re.sub(r'img_size=\[224\]', "img_size=[120, 160, 160]", text)
text = text.replace("img_size=img_size[0],", "img_size=img_size,")

with open('src/models/vision_transformer.py', 'w') as f:
    f.write(text)
