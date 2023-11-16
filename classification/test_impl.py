# from wintome_nat import WinToMeNATLevel, WinToMeNeighborhoodAttention2D

# from natten import NeighborhoodAttention2D as NeighborhoodAttention
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from typing import Tuple


from torchsummary import summary
import torch
import sys

from tome.merge import (
    attentive_bipartite_matching, linear_interpolation
)


def trap(message=""):
    sys.exit(message)

class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        preserved_attn = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x


tomeattn = ToMeAttention(dim=25, num_heads=5)

a = tomeattn(torch.rand(4, 10, 25))

print(f"{a.shape = }")
# print(f"{k.shape = }")
# print(f"{attn.shape = }")

m, u = linear_interpolation(r=5)

out = m(a)
print(out.shape)
