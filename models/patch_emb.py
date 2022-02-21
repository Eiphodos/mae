import torch
import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """ Volume to Patch Embedding
    """

    def __init__(self, vol_size=36, patch_size=6, in_chans=1, embed_dim=768):
        super().__init__()
        self.grid_size =((vol_size[0] // patch_size[0]), (vol_size[1] // patch_size[1]), (vol_size[2] // patch_size[2]))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.vol_size = vol_size
        self.patch_size = patch_size

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert x.shape[2] == self.vol_size[0]
        assert x.shape[3] == self.vol_size[1]
        assert x.shape[4] == self.vol_size[2]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        assert x.shape[1] == self.num_patches
        return x
