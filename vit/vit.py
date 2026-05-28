"""
Vision Transformer (ViT) 本体

論文: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
     Dosovitskiy et al., ICLR 2021  https://arxiv.org/abs/2010.11929

全体の流れ:
    (B, C, H, W)
        ↓  ① PatchEmbedding     → (B, N, D)
        ↓  ② CLS token 付加     → (B, N+1, D)
        ↓  ③ Positional Embed   → (B, N+1, D)
        ↓  ④ Transformer ×L     → (B, N+1, D)
        ↓  ⑤ CLS token 取り出し → (B, D)
        ↓  ⑥ MLP Head           → (B, num_classes)
"""

import torch
import torch.nn as nn

from .layer_norm import LayerNorm
from .patch_embedding import PatchEmbedding
from .transformer import TransformerEncoderBlock


class ViT(nn.Module):
    """
    Args:
        image_size  : 入力画像の一辺 (default: 32 for CIFAR-10)
        patch_size  : パッチの一辺   (default: 4 → 8×8=64 patches)
        in_channels : 入力チャネル数 (default: 3)
        num_classes : 分類クラス数   (default: 10)
        embed_dim   : 埋め込み次元   (default: 256)
        depth       : Encoder Block の層数 (default: 6)
        num_heads   : MHSA のヘッド数     (default: 8)
        mlp_dim     : FFN の隠れ層次元    (default: 512)
        dropout     : Attention/FFN の Dropout 率
        emb_dropout : Embedding 直後の Dropout 率
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── ① Patch Embedding ────────────────────────────────────────────
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches  # (32/4)^2 = 64

        # ── ② CLS Token ──────────────────────────────────────────────────
        # BERT と同様に学習可能な 1 トークンを先頭に付加
        # forward() で expand() してバッチ次元を合わせる
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # ── ③ Positional Embedding ───────────────────────────────────────
        # [CLS] + N patches = N+1 個のトークン分の位置情報を学習
        self.pos_embed  = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        # ── ④ Transformer Encoder ────────────────────────────────────────
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # ── ⑤⑥ Classification Head ───────────────────────────────────────
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 重みの初期化
        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """ViT 原論文に倣った重み初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # CLS token と pos_embed は truncated normal で初期化し直す
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # ① Patch Embedding: (B, C, H, W) → (B, N, D)
        x = self.patch_embed(x)

        # ② CLS token をバッチ数分コピーして先頭に結合
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)            # (B, N+1, D)

        # ③ Positional Embedding を加算 + Dropout
        x = x + self.pos_embed                           # broadcast over B
        x = self.emb_dropout(x)

        # ④ Transformer Encoder（L ブロック）
        x = self.transformer(x)                          # (B, N+1, D)

        # ⑤ 最終 LayerNorm → CLS token を取り出す
        x = self.norm(x)
        cls_out = x[:, 0]                                # (B, D)

        # ⑥ 分類ヘッド
        return self.head(cls_out)                        # (B, num_classes)

    # ─────────────────────────────────────────────────────────────────────
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
