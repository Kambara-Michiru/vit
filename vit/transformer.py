"""
Transformer Encoder Block モジュール

Pre-LN 構成（原論文の Appendix B で推奨）:
    x → LayerNorm → MHSA → (+x) → LayerNorm → FFN → (+x)

入力  : (B, N, D)
出力  : (B, N, D)
"""

import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .layer_norm import LayerNorm


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN)

    Linear → GELU → Dropout → Linear → Dropout
    """

    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),                      # ViT 原論文は GELU を使用
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    ViT の 1 Encoder ブロック（Pre-LN）

    Args:
        embed_dim : 埋め込み次元数
        num_heads : アテンションヘッド数
        mlp_dim   : FFN の隠れ層次元数（通常 embed_dim * 4 程度）
        dropout   : Attention / FFN の Dropout 率
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.ff    = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        # Self-Attention ブランチ（残差付き）
        x = x + self.attn(self.norm1(x))
        # Feed-Forward ブランチ（残差付き）
        x = x + self.ff(self.norm2(x))
        return x
