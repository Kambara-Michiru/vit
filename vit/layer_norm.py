"""
Layer Normalization スクラッチ実装

論文: "Layer Normalization" Ba et al., 2016
     https://arxiv.org/abs/1607.06450

## 数式
    y = gamma * (x - mean) / sqrt(var + eps) + beta

    - mean, var : 正規化する次元(normalized_shape)に沿って計算
    - gamma (weight) : スケール パラメータ（学習可能、初期値 1）
    - beta  (bias)   : シフト パラメータ（学習可能、初期値 0）
    - eps           : ゼロ除算防止の微小定数（default: 1e-5）

## BatchNorm との違い
    BatchNorm : バッチ次元 B に沿って正規化  → バッチサイズ依存、推論時に移動平均が必要
    LayerNorm : 特徴次元 D に沿って正規化   → バッチサイズ非依存、Transformer に適合

## ViT での使われ方
    入力テンソルは (B, N, D) 形状。
    normalized_shape = D なので、各トークンベクトル（長さ D）を独立に正規化する。
    バッチ B とトークン位置 N をまたいで統計量を共有しない点が重要。
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Args:
        normalized_shape : 正規化する次元のサイズ。int または tuple。
                           int を渡すと最終 1 次元のみ正規化（ViT の通常用途）。
        eps              : 分散の分母に加える安定化定数（default: 1e-5）
        elementwise_affine: True のとき weight / bias を学習する（default: True）
    """

    def __init__(
        self,
        normalized_shape: int | tuple,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()

        # int を渡された場合は 1 次元タプルに統一
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            # gamma : スケール（初期値 1）
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            # beta  : シフト（初期値 0）
            self.bias   = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            # affine なしの場合は登録しない
            self.register_parameter("weight", None)
            self.register_parameter("bias",   None)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 正規化する次元インデックスを末尾から数えて決める
        # 例: normalized_shape=(256,) なら dims=(-1,)
        #     normalized_shape=(4,256) なら dims=(-2,-1)
        ndim = len(self.normalized_shape)
        dims = tuple(range(-ndim, 0))

        # ── 平均・分散を正規化次元に沿って計算 ──────────────────────────
        mean = x.mean(dim=dims, keepdim=True)            # (B, N, 1)
        # unbiased=False : 母分散（PyTorch の nn.LayerNorm と同じ）
        var  = x.var(dim=dims, keepdim=True, unbiased=False)  # (B, N, 1)

        # ── 正規化 ───────────────────────────────────────────────────────
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # (B, N, D)

        # ── Affine 変換 (gamma * x + beta) ──────────────────────────────
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm

    # ─────────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
