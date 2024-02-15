import mock
import torch
from segment_anything.modeling.image_encoder import (
    Block,
    PatchEmbed,
    window_partition,
    window_unpartition,
)


def Block__forward(self, x: torch.Tensor) -> torch.Tensor:
    shortcut = x
    x = self.norm1(x)
    # Window partition
    H, W = 0, 0
    pad_hw = (0, 0)
    if self.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)

    x = self.attn(x)
    # Reverse window partition
    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))

    x = shortcut + x
    x = x + self.mlp(self.norm2(x))

    return x


def PatchEmbed__forward(self, x: torch.Tensor) -> torch.Tensor:
    #x = self.proj(x.to(torch.float32).squeeze(1))
    x = self.proj(x.squeeze(1))
    # B C H W -> B H W C
    x = x.permute(0, 2, 3, 1)
    return x


patches = (
    mock.patch.object(Block, "forward", Block__forward),
    mock.patch.object(PatchEmbed, "forward", PatchEmbed__forward),
)
