"""
3 本の Jupyter Notebook を生成するスクリプト。
  python build_notebooks.py
で 01_modules.ipynb / 02_attention_maps.ipynb / 03_pos_embedding.ipynb を作成する。
"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent

# ── ヘルパー ─────────────────────────────────────────────────────────────────

def md(src: str):
    return nbf.v4.new_markdown_cell(src)

def code(src: str):
    return nbf.v4.new_code_cell(src)

def save(nb, name: str):
    path = HERE / name
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"✅ {path}")

# ── 共通ヘッダ（全ノートブックの先頭に置く） ─────────────────────────────────
COMMON_SETUP = """\
import sys, pathlib
# notebooks/ の親（vit/）を Python パスに追加して vit パッケージを import できるようにする
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import matplotlib
import matplotlib.font_manager as _fm
# macOS 標準の日本語フォントを自動検出して設定（なければ DejaVu にフォールバック）
_jp = next((f.name for f in _fm.fontManager.ttflist if "Hiragino Sans" == f.name), None)
if _jp:
    matplotlib.rcParams["font.family"] = _jp
matplotlib.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け防止

import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from viz_utils import (
    CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD,
    denormalize, to_hwc, show_image, show_patches,
    overlay_attention, attention_rollout,
)
from vit import ViT

MODEL_PATH  = "../best_vit.pth"
DATA_ROOT   = "../data"
PATCH_SIZE  = 4
EMBED_DIM   = 256
DEPTH       = 6
NUM_HEADS   = 8
MLP_DIM     = 512
"""

LOAD_MODEL = """\
model = ViT(
    image_size=32, patch_size=PATCH_SIZE, num_classes=10,
    embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS,
    mlp_dim=MLP_DIM, dropout=0.1, emb_dropout=0.1,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print(f"モデルロード完了  パラメータ数: {model.num_parameters():,}")
"""

LOAD_DATA = """\
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_ds = torchvision.datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=False, transform=transform
)

# クラスごとに 1 枚ずつ集める
by_class: dict[int, torch.Tensor] = {}
for img, label in test_ds:
    if label not in by_class:
        by_class[label] = img
    if len(by_class) == 10:
        break

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for cls_id, ax in zip(range(10), axes.flat):
    show_image(by_class[cls_id], ax=ax, title=CIFAR10_CLASSES[cls_id])
fig.suptitle("CIFAR-10 テスト画像（各クラス 1 枚）", fontsize=13)
plt.tight_layout()
plt.show()
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 01 — モジュール形状追跡
# ═══════════════════════════════════════════════════════════════════════════════

NB01_CELLS = [
    md("""\
# 01 — モジュール形状追跡

ViT の各モジュールを通過するたびにテンソルの**形状**と**値**がどう変わるかを
1 ステップずつ追いかける。学習済みモデルを使うことで「意味のある表現」を観察できる。

| ステップ | 形状 | 概要 |
|---|---|---|
| 入力画像 | `(1, 3, 32, 32)` | CIFAR-10 |
| PatchEmbedding | `(1, 64, 256)` | 64 パッチ × 256 次元 |
| CLS token 付加 | `(1, 65, 256)` | 先頭に 1 分類トークン |
| Positional Embedding | `(1, 65, 256)` | 加算なので形は不変 |
| Transformer ×6 | `(1, 65, 256)` | 形は不変、内容が変化 |
| CLS 取り出し | `(256,)` | 分類用の代表ベクトル |
| Head | `(10,)` | クラス確率 |
"""),

    code(COMMON_SETUP + LOAD_MODEL),
    md("## テスト画像をロード"),
    code(LOAD_DATA),

    # ─── ① PatchEmbedding ───────────────────────────────────────────────
    md("""\
## ① Patch Embedding

32×32 の画像を **4×4 のパッチに分割**し、各パッチ（=48 次元ベクトル）を
線形射影で **256 次元**に変換する。
64 枚のパッチ → 64 個の 256 次元ベクトル列。
"""),

    code("""\
# 「frog」を使って追ってみる
img = by_class[6]   # frog

# 元画像とパッチグリッドを並べて表示
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
show_image(img, ax=axes[0], title="元画像 (32×32 px)")

# パッチを 8×8 グリッドに並べて表示
import torchvision.utils as vutils
from einops import rearrange

patches = rearrange(
    denormalize(img),
    "c (h p1) (w p2) -> (h w) c p1 p2",
    p1=PATCH_SIZE, p2=PATCH_SIZE,
)  # (64, 3, 4, 4)

grid = vutils.make_grid(patches, nrow=8, padding=1, normalize=False)
axes[1].imshow(grid.permute(1, 2, 0).numpy(), interpolation="nearest")
axes[1].set_title(f"8×8 パッチグリッド（{patches.shape[0]} patches）")
axes[1].axis("off")
plt.tight_layout()
plt.show()

print(f"1 パッチのサイズ: {PATCH_SIZE}×{PATCH_SIZE}×3 = {PATCH_SIZE*PATCH_SIZE*3} 次元  →  線形射影  →  {EMBED_DIM} 次元")
"""),

    code("""\
# PatchEmbedding の入出力形状を確認
with torch.no_grad():
    x_raw = img.unsqueeze(0)              # (1, 3, 32, 32)
    x_emb = model.patch_embed(x_raw)     # (1, 64, 256)

print("─── PatchEmbedding 形状変化 ─────────────────────────────────")
print(f"  入力 : {tuple(x_raw.shape)}        (B, C, H, W)")
print(f"  出力 : {tuple(x_emb.shape)}  (B, N_patches, embed_dim)")
print()
print("各パッチベクトルの統計:")
print(f"  mean = {x_emb.mean():.4f},  std = {x_emb.std():.4f}")
print(f"  norm per patch (mean): {x_emb[0].norm(dim=-1).mean():.4f}")
"""),

    # ─── ② CLS Token ────────────────────────────────────────────────────
    md("""\
## ② CLS Token 付加

BERT の `[CLS]` と同じ発想。
学習可能な **1 ベクトル**をシーケンスの先頭に付加する。
このトークンが Transformer を通じて画像全体の情報を集約し、最終的に分類に使われる。
"""),

    code("""\
with torch.no_grad():
    cls = model.cls_token.expand(1, -1, -1)       # (1,  1, 256)
    x_with_cls = torch.cat([cls, x_emb], dim=1)  # (1, 65, 256)

print("─── CLS token 付加 ──────────────────────────────────────────")
print(f"  パッチ列     : {tuple(x_emb.shape)}")
print(f"  CLS token   : {tuple(cls.shape)}")
print(f"  結合後       : {tuple(x_with_cls.shape)}  ← index 0 が CLS")

# 全 65 トークンのノルムを棒グラフで可視化
norms = x_with_cls[0].norm(dim=-1).numpy()   # (65,)
colors = ["tomato"] + ["steelblue"] * 64

fig, ax = plt.subplots(figsize=(13, 2.8))
ax.bar(range(65), norms, color=colors, width=0.8)
ax.set_xlabel("Token index  (0 = CLS、1–64 = patches)")
ax.set_ylabel("L2 ノルム")
ax.set_title("各トークンの L2 ノルム（赤=CLS、青=patch）")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()
"""),

    # ─── ③ Positional Embedding ─────────────────────────────────────────
    md("""\
## ③ Positional Embedding

全 65 トークンに「自分はシーケンスの何番目か」という位置情報を**加算**する。
重みは学習によって獲得される（ViT は固定 sin/cos ではなく学習可能）。
"""),

    code("""\
with torch.no_grad():
    x_before_pos = x_with_cls.clone()
    x_after_pos  = x_with_cls + model.pos_embed   # (1, 65, 256)

# 位置埋め込みそのもののノルム（各トークンへの「追加情報量」）
pos_norms = model.pos_embed[0].detach().norm(dim=-1).numpy()   # (65,)

fig, axes = plt.subplots(1, 2, figsize=(13, 3))

# 加算前後のノルム比較
axes[0].plot(x_before_pos[0].norm(dim=-1).numpy(), label="加算前", alpha=0.8)
axes[0].plot(x_after_pos [0].norm(dim=-1).numpy(), label="加算後", alpha=0.8)
axes[0].set_xlabel("Token index")
axes[0].set_ylabel("L2 ノルム")
axes[0].set_title("Positional Embedding 加算前後のノルム比較")
axes[0].legend()

# pos_embed 自体のノルムを 8×8 マップとして表示（パッチ部分のみ）
pos_patch_norms = pos_norms[1:].reshape(8, 8)  # CLS を除いた 64 パッチ分
im = axes[1].imshow(pos_patch_norms, cmap="viridis", interpolation="nearest")
axes[1].set_title("位置埋め込みのノルム（8×8 パッチグリッド）")
axes[1].set_xlabel("パッチ列")
axes[1].set_ylabel("パッチ行")
plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.show()
"""),

    # ─── ④ Transformer Encoder ─────────────────────────────────────────
    md("""\
## ④ Transformer Encoder（6 層）

各層で Self-Attention + FFN を適用し、パッチ間の依存関係を学習する。
CLS トークンのノルムを層ごとにプロットして、表現がどう成長するかを観察する。
"""),

    code("""\
with torch.no_grad():
    x = x_after_pos.clone()
    x = model.emb_dropout(x)

    cls_norms   = []   # CLS ノルムの推移
    patch_norms = []   # パッチ平均ノルムの推移

    for i, block in enumerate(model.transformer):
        x = block(x)
        cn = x[0, 0].norm().item()
        pn = x[0, 1:].norm(dim=-1).mean().item()
        cls_norms.append(cn)
        patch_norms.append(pn)
        print(f"  Block {i}:  CLS norm = {cn:.4f}  |  patch mean norm = {pn:.4f}")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(range(DEPTH), cls_norms,   "o-", color="tomato",    lw=2, ms=8, label="CLS")
ax.plot(range(DEPTH), patch_norms, "s-", color="steelblue", lw=2, ms=8, label="patches (mean)")
for i, (c, p) in enumerate(zip(cls_norms, patch_norms)):
    ax.annotate(f"{c:.2f}", (i, c), xytext=(0, 8),  textcoords="offset points",
                ha="center", fontsize=8, color="tomato")
    ax.annotate(f"{p:.2f}", (i, p), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=8, color="steelblue")
ax.set_xlabel("Transformer Block")
ax.set_ylabel("L2 ノルム")
ax.set_title("CLS / パッチ トークンのノルム推移（層が深まるにつれ表現が形成される）")
ax.set_xticks(range(DEPTH))
ax.set_xticklabels([f"Block {i}" for i in range(DEPTH)])
ax.legend()
plt.tight_layout()
plt.show()
"""),

    # ─── ⑤ Classification Head ─────────────────────────────────────────
    md("""\
## ⑤ Classification Head

Transformer 出力の **CLS トークン**（index=0）だけを取り出して
Linear(256 → 10) → Softmax でクラス確率を出力する。
"""),

    code("""\
with torch.no_grad():
    x_normed  = model.norm(x)         # LayerNorm
    cls_vec   = x_normed[0, 0]        # (256,)  ← CLS トークン
    logits    = model.head(cls_vec)   # (10,)
    probs     = torch.softmax(logits, dim=-1)

pred = probs.argmax().item()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

show_image(img, ax=axes[0], title="入力: frog")

colors = ["tomato" if i == pred else "steelblue" for i in range(10)]
bars = axes[1].barh(CIFAR10_CLASSES, probs.numpy(), color=colors)
axes[1].set_xlabel("確率")
axes[1].set_xlim(0, 1)
axes[1].set_title(f"予測: {CIFAR10_CLASSES[pred]}  ({probs[pred]*100:.1f}%)")
axes[1].invert_yaxis()
# 各バーに確率を表示
for bar, p in zip(bars, probs):
    axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{p*100:.1f}%", va="center", fontsize=8)
plt.tight_layout()
plt.show()
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 02 — Attention Map 可視化
# ═══════════════════════════════════════════════════════════════════════════════

NB02_CELLS = [
    md("""\
# 02 — Attention Map 可視化

学習済みモデルの **Attention weight** を hook で取り出して可視化する。

- **ヘッド別マップ**: 各ヘッドが何に注目しているか
- **CLS アテンション**: 分類に使われる CLS トークンがどこを見ているか
- **Attention Rollout**: 全層を合成して「最終的にどこを見た」かを 1 枚で表す
"""),

    code(COMMON_SETUP + LOAD_MODEL + LOAD_DATA),

    # ─── Hook の登録 ────────────────────────────────────────────────────
    md("""\
## Hook でアテンション重みを取り出す

`block.attn.attn_drop` は softmax 直後の Dropout モジュール。
ここへの **forward hook** で `(B, H, N, N)` のアテンション重みをキャプチャできる。
"""),

    code("""\
attn_store: dict[str, torch.Tensor] = {}

def make_hook(key: str):
    def hook(module, input, output):
        # input[0] = softmax 後のアテンション重み (B, H, N, N)
        attn_store[key] = input[0].detach().cpu()
    return hook

handles = []
for i, block in enumerate(model.transformer):
    h = block.attn.attn_drop.register_forward_hook(make_hook(f"layer_{i}"))
    handles.append(h)

print(f"{len(handles)} 個の hook を登録しました")
print("キャプチャされるキー:", [f"layer_{i}" for i in range(len(handles))])
"""),

    # ─── 推論実行 ────────────────────────────────────────────────────────
    md("""\
## 推論の実行

画像を 1 枚選んで forward を走らせる。
hook が発火してアテンション重みが `attn_store` に保存される。
"""),

    code("""\
# 「bird」で試す
TARGET_CLASS = 2   # bird
img = by_class[TARGET_CLASS]

with torch.no_grad():
    logits = model(img.unsqueeze(0))     # forward → hook が発火
    probs  = torch.softmax(logits[0], dim=-1)

pred = probs.argmax().item()

fig, ax = plt.subplots(figsize=(3, 3))
show_image(img, ax=ax,
           title=f"正解: {CIFAR10_CLASSES[TARGET_CLASS]}\\n"
                 f"予測: {CIFAR10_CLASSES[pred]} ({probs[pred]*100:.1f}%)")
plt.tight_layout()
plt.show()

# attn_store の内容確認
for k, v in attn_store.items():
    print(f"  {k}: {tuple(v.shape)}  (B, H, N, N)")
"""),

    # ─── ヘッド別マップ（最終層） ────────────────────────────────────────
    md("""\
## ヘッド別アテンションマップ（最終層）

最終層（layer_5）の 8 ヘッドそれぞれが「どのトークンからどのトークンへ注目しているか」を表示する。
行 = query トークン、列 = key トークン。明るいほど注目度が高い。
"""),

    code("""\
LAYER = DEPTH - 1  # 最終層
attn = attn_store[f"layer_{LAYER}"][0]  # (H, 65, 65) ← バッチ次元を除去

fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
for h, ax in enumerate(axes.flat):
    im = ax.imshow(attn[h].numpy(), cmap="viridis", vmin=0)
    ax.set_title(f"Head {h}", fontsize=10)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query")
    # CLS トークンの境界線
    ax.axhline(0.5, color="red", lw=0.8)
    ax.axvline(0.5, color="red", lw=0.8)

fig.suptitle(f"Layer {LAYER} — 8 ヘッドのアテンションマップ (65×65)\\n"
             f"赤線の左上が CLS token", fontsize=12)
fig.colorbar(im, ax=axes, fraction=0.015, pad=0.04)
plt.show()
"""),

    # ─── CLS アテンション（全層） ────────────────────────────────────────
    md("""\
## CLS → パッチ アテンション（全層）

`attn[h, 0, 1:]` = CLS トークン（index=0）が各パッチ（index=1–64）へ払うアテンション。
これを 8×8 グリッドにリシェイプして元画像に重ね合わせる。
**層が深まるにつれて注目が特定の領域に絞られていく**様子が見える。
"""),

    code("""\
fig, axes = plt.subplots(2, DEPTH // 2, figsize=(14, 6))
axes = axes.flat

for layer_idx in range(DEPTH):
    attn = attn_store[f"layer_{layer_idx}"][0]  # (H, 65, 65)
    # ヘッド平均の CLS → パッチ アテンション
    cls_attn = attn.mean(dim=0)[0, 1:]          # (64,)

    overlay_attention(img, cls_attn, ax=axes[layer_idx],
                      title=f"Layer {layer_idx}  (head avg)")

fig.suptitle("CLS トークンのアテンション（全層・ヘッド平均）\\n"
             "明るい部分 = モデルが注目しているパッチ",
             fontsize=12)
plt.tight_layout()
plt.show()
"""),

    code("""\
# 最終層は各ヘッド個別にも表示
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
attn_last = attn_store[f"layer_{DEPTH-1}"][0]  # (H, 65, 65)

for h, ax in enumerate(axes.flat):
    cls_attn = attn_last[h, 0, 1:]   # (64,)
    overlay_attention(img, cls_attn, ax=ax, title=f"Layer {DEPTH-1} / Head {h}")

fig.suptitle(f"最終層（Layer {DEPTH-1}）の各ヘッド CLS アテンション", fontsize=12)
plt.tight_layout()
plt.show()
"""),

    # ─── Attention Rollout ───────────────────────────────────────────────
    md("""\
## Attention Rollout

全 6 層のアテンションを**積算合成**して「入力パッチから最終的に CLS トークンへ
どれだけ情報が流れたか」を 1 枚のマップで表す。
単層の CLS アテンションより安定した可視化が得られる。

> Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
"""),

    code("""\
# 全層のアテンション（ヘッド次元あり）をリストにまとめる
attn_per_layer = [
    attn_store[f"layer_{i}"][0]  # (H, 65, 65)
    for i in range(DEPTH)
]

rollout = attention_rollout(attn_per_layer)  # (64,)
print(f"Rollout の形状: {tuple(rollout.shape)}")
print(f"最大注意パッチ index: {rollout.argmax().item()}  "
      f"(8×8 grid 上: row={rollout.argmax().item()//8}, col={rollout.argmax().item()%8})")

fig, axes = plt.subplots(1, 3, figsize=(11, 4))

show_image(img, ax=axes[0], title=f"入力画像\\n({CIFAR10_CLASSES[TARGET_CLASS]})")

# 最終層のヘッド平均 CLS アテンション
cls_last = attn_store[f"layer_{DEPTH-1}"][0].mean(0)[0, 1:]
overlay_attention(img, cls_last, ax=axes[1],
                  title=f"最終層 CLS アテンション\\n(head avg)")

# Rollout
overlay_attention(img, rollout, ax=axes[2],
                  title="Attention Rollout\\n(全層統合)")

fig.suptitle("CLS アテンション: 最終層 vs Rollout", fontsize=12)
plt.tight_layout()
plt.show()
"""),

    code("""\
# hook を解除
for h in handles:
    h.remove()
print("hook を解除しました")
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 03 — Positional Embedding 可視化
# ═══════════════════════════════════════════════════════════════════════════════

NB03_CELLS = [
    md("""\
# 03 — Positional Embedding 可視化

モデルが「どこにあるか」を学習した**位置埋め込み**の内部構造を可視化する。

- **コサイン類似度マトリクス**: 65 トークン間の類似度（隣接パッチが近ければ位置が学習できている）
- **各位置の類似度マップ**: ある 1 つのパッチと他パッチとの類似度を 8×8 マップで表示
- **PCA 2D プロット**: 256 次元の位置埋め込みを 2D に圧縮し、格子状に並ぶかを確認
"""),

    code(COMMON_SETUP + LOAD_MODEL),

    # ─── 位置埋め込みの取り出し ──────────────────────────────────────────
    md("""\
## 位置埋め込みを取り出す

`model.pos_embed` は `(1, 65, 256)` のパラメータ。
- `pos_embed[0, 0]`    : CLS token の位置埋め込み
- `pos_embed[0, 1:65]` : パッチ 0–63 の位置埋め込み（row-major, 8×8 順）
"""),

    code("""\
pos_embed = model.pos_embed[0].detach()  # (65, 256)

cls_pos   = pos_embed[0]     # (256,)   CLS
patch_pos = pos_embed[1:]    # (64, 256) パッチ

print(f"pos_embed: {tuple(pos_embed.shape)}")
print(f"  CLS  pos: {tuple(cls_pos.shape)},  norm = {cls_pos.norm():.4f}")
print(f"  patch pos (mean norm): {patch_pos.norm(dim=-1).mean():.4f}")
"""),

    # ─── コサイン類似度マトリクス ────────────────────────────────────────
    md("""\
## コサイン類似度マトリクス

全 65 トークンの位置埋め込み間のコサイン類似度を 65×65 のヒートマップで表す。
- 対角が 1.0 になるのは自分自身との類似度
- **隣接パッチが近い色**なら、位置情報が空間的に学習できている証拠
"""),

    code("""\
import torch.nn.functional as F

# コサイン類似度: (65, 65)
pos_norm = F.normalize(pos_embed, dim=-1)
sim = (pos_norm @ pos_norm.T).numpy()   # (65, 65)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(sim, cmap="RdYlBu_r", vmin=-1, vmax=1, interpolation="nearest")
ax.set_title("位置埋め込みのコサイン類似度マトリクス (65×65)\\n"
             "index 0 = CLS、index 1–64 = patches (row-major)", fontsize=11)
ax.set_xlabel("Token index")
ax.set_ylabel("Token index")
# CLS 境界を強調
ax.axhline(0.5, color="white", lw=1.5)
ax.axvline(0.5, color="white", lw=1.5)
ax.text(0, 0, "CLS", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
plt.colorbar(im, fraction=0.04, pad=0.03)
plt.tight_layout()
plt.show()

# 上位類似ペアを確認（CLS を除く）
sim_patch = sim[1:, 1:]   # (64, 64)
print("パッチ間コサイン類似度の統計:")
print(f"  mean = {sim_patch.mean():.4f}")
print(f"  std  = {sim_patch.std():.4f}")
print(f"  隣接パッチ（左右 1 ステップ）の平均類似度 ≈ "
      f"{sim_patch[range(0, 56), range(1, 57)].mean():.4f}")
"""),

    # ─── 各位置の類似度マップ ─────────────────────────────────────────────
    md("""\
## 各パッチ位置の類似度マップ

あるパッチと他の全パッチとのコサイン類似度を 8×8 マップとして可視化する。
**空間的に連続した構造**があれば位置埋め込みが正しく学習できている。
"""),

    code("""\
# 4 つの代表的な位置を選ぶ
positions = {
    "左上 (0,0)":   0,
    "右上 (0,7)":   7,
    "中央 (3,3)":  27,
    "右下 (7,7)":  63,
}

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (label, idx) in zip(axes, positions.items()):
    sim_row = sim[idx + 1, 1:].reshape(8, 8)   # +1 は CLS offset
    im = ax.imshow(sim_row, cmap="RdYlBu_r", vmin=-0.5, vmax=1.0,
                   interpolation="nearest")
    # 基準パッチの位置に × 印
    row, col = divmod(idx, 8)
    ax.plot(col, row, "wx", markersize=12, markeredgewidth=2.5)
    ax.set_title(f"{label}\\n(patch {idx})", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.06)

fig.suptitle("各パッチと他パッチとのコサイン類似度マップ\\n"
             "（×=基準パッチ、周囲ほど明るい = 空間的な位置が学習できている）",
             fontsize=12)
plt.tight_layout()
plt.show()
"""),

    # ─── PCA ────────────────────────────────────────────────────────────
    md("""\
## PCA で 2D 可視化

64 パッチの位置埋め込み（256 次元）を PCA で 2 次元に圧縮する。
うまく学習できていれば **8×8 の格子状パターン**が現れるはず。
"""),

    code("""\
# PyTorch の pca_lowrank を使う（scikit-learn 不要）
U, S, V = torch.pca_lowrank(patch_pos, q=2)
coords = (patch_pos @ V).numpy()  # (64, 2)

fig, ax = plt.subplots(figsize=(7, 7))

# 8×8 グリッドの行・列ごとに色を付ける
row_ids = torch.arange(64) // 8   # 0–7
col_ids = torch.arange(64) %  8   # 0–7

scatter = ax.scatter(coords[:, 0], coords[:, 1],
                     c=row_ids.numpy(), cmap="tab10",
                     s=120, zorder=3)
# パッチ番号のラベル
for idx, (x, y) in enumerate(coords):
    ax.annotate(str(idx), (x, y), fontsize=7,
                ha="center", va="center", color="white", fontweight="bold")

# 格子の辺（行が同じパッチを線でつなぐ）
for r in range(8):
    row_mask = (row_ids == r).numpy()
    row_coords = coords[row_mask]
    # 列順にソート
    order = col_ids[row_ids == r].numpy().argsort()
    ax.plot(row_coords[order, 0], row_coords[order, 1],
            "-", color="gray", alpha=0.4, lw=0.8)

plt.colorbar(scatter, ax=ax, label="行インデックス (0=上、7=下)")
ax.set_title("位置埋め込みの PCA 2D 可視化\\n"
             "（色=行, 数字=パッチ番号, 格子状に並ぶほど位置が学習されている）",
             fontsize=11)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
plt.tight_layout()
plt.show()

explained = (S[:2] ** 2 / (S ** 2).sum() * 100)
print(f"PC1 説明分散率: {explained[0]:.1f}%")
print(f"PC2 説明分散率: {explained[1]:.1f}%")
print(f"PC1+PC2 合計  : {explained.sum():.1f}%")
"""),

    md("""\
## まとめ

| 観察 | 意味 |
|---|---|
| 類似度マトリクスで隣接パッチが高い相関 | モデルが空間的な近さを学習できている |
| 類似度マップで中心からの距離勾配 | 位置情報が連続的 |
| PCA プロットが格子状 | 埋め込みが 2D 空間の構造を保持している |
| CLS が全パッチと独立した低類似度 | CLS は特殊な「集約用」トークンとして機能している |
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 生成・保存
# ═══════════════════════════════════════════════════════════════════════════════

def build():
    for name, cells in [
        ("01_modules.ipynb",       NB01_CELLS),
        ("02_attention_maps.ipynb", NB02_CELLS),
        ("03_pos_embedding.ipynb",  NB03_CELLS),
    ]:
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        save(nb, name)

if __name__ == "__main__":
    build()
    print("\n--- 完了 ---")
    print("起動方法: jupyter lab  （notebooks/ ディレクトリで実行）")
