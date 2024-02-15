from typing import Tuple
import mock
import torch
import numpy as np
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom


def forward(self, size: Tuple[int, int]) -> torch.Tensor:
    """Generate positional encoding for a grid of the specified size."""
    h, w = size
    device = self.positional_encoding_gaussian_matrix.device
    grid = torch.ones((h, w), device=device, dtype=torch.float32)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w

    pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
    return pe.permute(2, 0, 1)  # C x H x W

def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
    """Positionally encode points that are normalized to [0,1]."""
    # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
    coords = 2 * coords - 1
    coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
    coords = 2 * np.pi * coords
    # outputs d_1 x ... x d_n x C shape
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

patches = (mock.patch.object(PositionEmbeddingRandom, "forward", forward),
           mock.patch.object(PositionEmbeddingRandom, "_pe_encoding", _pe_encoding))