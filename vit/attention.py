"""
Multi-Head Self-Attention (MHSA) モジュール

入力  : (B, N, D)
出力  : (B, N, D)

Scaled Dot-Product Attention:
    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

## Q/K/V の形状変換（2ステップ）
    (B, N, D)
      → reshape(B, N, H, head_dim)   # D を「ヘッド数 × head_dim」に分解
      → transpose(1, 2)              # H を先頭へ移動 → (B, H, N, head_dim)
                                     # こうすることで @ の最後2次元が
                                     # (N, head_dim) となり、H ヘッドを
                                     # バッチとして一括行列積できる

    ※ reshape だけでは (B, N, H, head_dim) のまま H が内側にとどまり、
      @-演算子で (N×N) の attention map を H 個同時に計算できない。
      transpose でバッチ次元 (B, H) を揃えるのが重要。

## fused QKV について
    疑似コードは fc_q/fc_k/fc_v の 3 本の Linear を使うが、
    1 本の Linear(D → 3D) にまとめると GEMM が 1 回で済み GPU 効率が高い。
    出力を chunk(3) で均等分割すれば数学的に等価。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Args:
        embed_dim : モデルの次元数 D
        num_heads : アテンションヘッド数 H（embed_dim % num_heads == 0 が必要）
        dropout   : Attention weight への Dropout 率
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads      # d_k = D / H
        self.scale = self.head_dim ** -0.5          # 1 / sqrt(d_k)

        # Q, K, V を 1 本の Linear で一括計算（fused: GEMM が 1 回で済む）
        # bias=False は ViT 原論文の設定
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # 出力射影: ヘッドを統合し情報を再混合する重要な線形変換
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # ─ Q / K / V の計算 ─────────────────────────────────────────────
        # fused linear → (B, N, 3*D) → 均等に 3 分割
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # それぞれ (B, N, D)

        # ─ マルチヘッド形状へ変換 ────────────────────────────────────────
        # ① reshape: D を (H, head_dim) に分解 → (B, N, H, head_dim)
        # ② transpose(1, 2): H を前へ          → (B, H, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # ─ Scaled Dot-Product Attention ─────────────────────────────────
        # (B, H, N, head_dim) @ (B, H, head_dim, N) → (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # ─ 重み付き和 ────────────────────────────────────────────────────
        # (B, H, N, N) @ (B, H, N, head_dim) → (B, H, N, head_dim)
        out = attn @ v

        # ─ ヘッドを結合して元の形状に戻す ───────────────────────────────
        # ① transpose(1, 2): (B, N, H, head_dim)
        # ② reshape: H と head_dim を結合 → (B, N, D)
        #    ※ transpose 後は非連続テンソルになるため reshape が内部で
        #      contiguous コピーを行う（.contiguous().view() と等価）
        out = out.transpose(1, 2).reshape(B, N, D)

        # ─ 出力射影 ──────────────────────────────────────────────────────
        return self.proj(out)
