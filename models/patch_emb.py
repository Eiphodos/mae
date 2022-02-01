import torch
import torch.nn as nn

class PatchEmbed3D(nn.Module):
    """ Volume to Patch Embedding
    """

    def __init__(self, vol_size=36, patch_size=6, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (vol_size // patch_size) * (vol_size // patch_size) * (vol_size // patch_size)
        self.vol_size = (vol_size, vol_size, vol_size)
        self.patch_size = (patch_size, patch_size, patch_size)
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x