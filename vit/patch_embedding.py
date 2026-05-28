"""
Patch Embedding モジュール

画像を固定サイズのパッチに分割し、線形投影で D 次元ベクトル列に変換する。

入力  : (B, C, H, W)
出力  : (B, N, D)   N = (H/P)*(W/P),  D = embed_dim
"""

import torch.nn as nn
from einops import rearrange

from .layer_norm import LayerNorm


class PatchEmbedding(nn.Module):
    """
    Args:
        image_size  : 画像の一辺のサイズ（正方形を想定）
        patch_size  : パッチの一辺のサイズ
        in_channels : 入力チャネル数（RGB = 3）
        embed_dim   : 投影後の埋め込み次元数
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 例: (32/4)^2 = 64
        patch_dim = in_channels * patch_size * patch_size   # 例: 3*4*4 = 48

        # Pre-LayerNorm → Linear Projection（tintn 実装に倣い LN を先に置く）
        self.projection = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
        )

    def forward(self, x):
        # x : (B, C, H, W)
        # ─ パッチ分割 & 平坦化 ─────────────────────────────────────────────
        # 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)'
        #   H 方向を (h グリッド数, p1 パッチ高) に分解
        #   W 方向を (w グリッド数, p2 パッチ幅) に分解
        #   → (B, N, patch_dim)
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        # ─ 線形投影 ────────────────────────────────────────────────────────
        # (B, N, patch_dim) → (B, N, embed_dim)
        return self.projection(x)
