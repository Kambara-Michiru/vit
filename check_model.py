"""
モデル構造の動作確認スクリプト
学習前にテンソルの形状・パラメータ数・forward パスを検証する
"""

import torch
from vit import ViT

# ─── モデル構築 ───────────────────────────────────────────────────────────────
model = ViT(
    image_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=256,
    depth=6,
    num_heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
)

model.eval()

# ─── パラメータ数 ─────────────────────────────────────────────────────────────
total  = sum(p.numel() for p in model.parameters())
trainable = model.num_parameters()
print(f"Total params     : {total:,}")
print(f"Trainable params : {trainable:,}")

# ─── 各モジュールのパラメータ数内訳 ──────────────────────────────────────────
print("\n── Module breakdown ──")
for name, module in model.named_children():
    n = sum(p.numel() for p in module.parameters())
    print(f"  {name:<20}: {n:>10,}")

# ─── Forward pass（バッチ=4, CIFAR-10 サイズ）────────────────────────────────
print("\n── Forward pass check ──")
dummy = torch.randn(4, 3, 32, 32)

with torch.no_grad():
    # 中間形状を確認するためステップごとに実行
    B = dummy.shape[0]

    # ① Patch Embedding
    x = model.patch_embed(dummy)
    print(f"  After PatchEmbedding : {tuple(x.shape)}  ← (B, N=64, D=256)")

    # ② CLS token 付加
    cls = model.cls_token.expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    print(f"  After CLS concat     : {tuple(x.shape)}  ← (B, N+1=65, D=256)")

    # ③ Positional Embedding
    x = x + model.pos_embed
    print(f"  After PosEmbed       : {tuple(x.shape)}  ← (B, 65, 256)")

    # ④ Transformer
    x = model.transformer(x)
    print(f"  After Transformer×6  : {tuple(x.shape)}  ← (B, 65, 256)")

    # ⑤⑥ Head
    x = model.norm(x)
    cls_out = x[:, 0]
    logits = model.head(cls_out)
    print(f"  After Head (logits)  : {tuple(logits.shape)}  ← (B, 10)")

# ─── 最終確認：full forward ───────────────────────────────────────────────────
with torch.no_grad():
    out = model(dummy)

print(f"\n✅ model(dummy) shape: {tuple(out.shape)}")
print(f"✅ logits sample: {out[0].tolist()[:5]} ...")
print("\nAll checks passed!")
