"""
ViT 可視化ユーティリティ
notebooks/ 内の全ノートブックから共通で import して使う。
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ── CIFAR-10 定数 ─────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ── 画像ユーティリティ ────────────────────────────────────────────────────

def denormalize(tensor):
    """CIFAR-10 正規化を元に戻す (C, H, W) → (C, H, W)"""
    t = tensor.clone().float()
    for c, (m, s) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        t[c] = t[c] * s + m
    return t.clamp(0, 1)


def to_hwc(tensor):
    """(C, H, W) テンソル → (H, W, C) numpy"""
    return denormalize(tensor).permute(1, 2, 0).numpy()


def show_image(img_tensor, ax=None, title=None):
    """(C, H, W) テンソルを 1 枚表示"""
    img = to_hwc(img_tensor)
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_patches(img_tensor, patch_size=4, figsize=(7, 7)):
    """
    画像を patch_size × patch_size のパッチグリッドに分解して可視化する。
    各パッチにインデックス番号（左上）を重ね合わせる。
    """
    img = denormalize(img_tensor)
    _, H, W = img.shape
    n = H // patch_size  # グリッドの一辺 (32/4 = 8)

    fig, axes = plt.subplots(n, n, figsize=figsize,
                             gridspec_kw=dict(wspace=0.04, hspace=0.04))
    for i in range(n):
        for j in range(n):
            patch = img[:,
                        i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size]
            ax = axes[i, j]
            ax.imshow(patch.permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.text(0.05, 0.88, str(i * n + j),
                    transform=ax.transAxes, fontsize=6,
                    color="white", fontweight="bold")
            ax.axis("off")

    fig.suptitle(
        f"Patch split: {n}×{n} = {n * n} patches  (each {patch_size}×{patch_size} px)",
        fontsize=11,
    )
    return fig


# ── アテンション可視化 ────────────────────────────────────────────────────

def overlay_attention(img_tensor, attn_1d, ax, title=None,
                      alpha=0.55, cmap="hot"):
    """
    1D アテンション (n_patches,) を 2D にアップスケールして画像に重ね合わせる。

    attn_1d : (n_patches,) = (64,) のテンソル。row-major で並んでいる前提。
    """
    img = to_hwc(img_tensor)
    H, W = img.shape[:2]
    n = int(round(len(attn_1d) ** 0.5))  # 8

    attn_grid = attn_1d.float().reshape(1, 1, n, n)
    attn_up = F.interpolate(attn_grid, size=(H, W),
                            mode="bilinear", align_corners=False
                            ).squeeze().numpy()
    lo, hi = attn_up.min(), attn_up.max()
    attn_up = (attn_up - lo) / (hi - lo + 1e-8)  # 0–1 正規化

    ax.imshow(img, interpolation="nearest")
    ax.imshow(attn_up, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def attention_rollout(attn_per_layer):
    """
    Attention Rollout (Abnar & Zuidema, 2020)

    全層のアテンションを集積し、CLS トークンへの情報フローを 1 枚のマップで推定。

    Args:
        attn_per_layer : list of Tensor (H, N, N)  — H ヘッド, N トークン数
    Returns:
        Tensor (N_patches,)  — CLS の各パッチへのロールアウト注意
    """
    N = attn_per_layer[0].shape[-1]
    rollout = torch.eye(N)

    for attn in attn_per_layer:
        attn_avg = attn.mean(dim=0)                          # ヘッド平均 (N, N)
        attn_aug = attn_avg + torch.eye(N)                   # 残差接続を加味
        attn_aug = attn_aug / attn_aug.sum(-1, keepdim=True) # 行正規化
        rollout  = attn_aug @ rollout

    return rollout[0, 1:]  # CLS → 各パッチ: (N_patches,)
