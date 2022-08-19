import torch
import numpy as np
import cv2


def rollout(attentions, discard_ratio, head_fusion, token_idx):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    # mask = result[0, token_idx, :]
    mask = result[0, :, token_idx]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def show_mask_on_image(img, mask):
    # mask = (np.clip(mask, 0, mask.mean())) / (np.clip(mask, 0, mask.mean())).max()
    mask = cv2.resize(mask, (img.shape[0], img.shape[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    return cam


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9, patch_size=16):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.patch_size = patch_size
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, img_lab, bool_hinted_pos, gt, hint_coords):
        self.attentions = []
        with torch.no_grad():
            outputs = self.model(img_lab, bool_hinted_pos)
        B, C, H, W = img_lab.shape

        token_coords = hint_coords // self.patch_size
        img_rgb = gt[0].permute(1, 2, 0)
        np_img = np.array(img_rgb.cpu())[:, :, ::-1]
        rollout_masks = []
        for token_coord in token_coords:
            token_idx = (token_coord[0] * (H / self.patch_size) + token_coord[1]).long()
            rollout_mask = rollout(self.attentions, self.discard_ratio, self.head_fusion, token_idx)
            rollout_mask = show_mask_on_image(np_img, rollout_mask)
            rollout_masks.append(rollout_mask)
        return rollout_masks, outputs
