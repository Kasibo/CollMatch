import torch
from mmseg.ops import resize
import torch.nn.functional as F
# from torchvision.transforms.functional import resize


class BlockMaskGenerator:
    def __init__(self, mask_ratio=0.5, mask_block_size=32, inner_mask_ratio=0.5, inner_block_size=4):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
        self.inner_mask_ratio = inner_mask_ratio
        self.inner_block_size = inner_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape

        # 第一层：大块级别遮挡
        grid_H, grid_W = H // self.mask_block_size, W // self.mask_block_size
        mask_shape = (B, 1, grid_H, grid_W)
        coarse_mask = torch.rand(mask_shape, device=imgs.device) > self.mask_ratio  # 1=保留, 0=遮挡
        coarse_mask = coarse_mask.float()
        coarse_mask_resized = resize(coarse_mask, size=(H, W))

        # 第二层：在遮挡块（flag=0）内部进行更细粒度遮挡
        fine_mask = torch.ones_like(coarse_mask)
        fine_grid_H, fine_grid_W = H // self.inner_block_size, W // self.inner_block_size
        fine_mask_shape = (B, 1, fine_grid_H, fine_grid_W)
        fine_random_mask = torch.rand(fine_mask_shape, device=imgs.device) > self.inner_mask_ratio
        fine_random_mask = fine_random_mask.float()
        fine_random_mask_resized = resize(fine_random_mask, size=(H, W))

        # 仅对 coarse_mask = 0 的区域应用 fine_random_mask
        final_mask = (coarse_mask_resized > 0) | (fine_random_mask_resized > 0)
        final_mask = final_mask.float()

        return final_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask