"""
CIFAR-10 で ViT を学習するスクリプト

使い方:
    python train.py [--epochs 30] [--batch-size 128] [--lr 1e-3]

学習レシピ:
    - Optimizer : AdamW (weight_decay=0.05)
    - LR Schedule: Linear Warmup + Cosine Annealing
    - Regularize : Label Smoothing(0.1), Dropout(0.1), Grad Clip
    - Augment    : RandomCrop + RandomHorizontalFlip
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from vit import ViT

# ─── CLI 引数 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train ViT on CIFAR-10")
parser.add_argument("--epochs",      type=int,   default=30)
parser.add_argument("--batch-size",  type=int,   default=128)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--weight-decay",type=float, default=0.05)
parser.add_argument("--warmup-epochs",type=int,  default=5)
parser.add_argument("--embed-dim",   type=int,   default=256)
parser.add_argument("--depth",       type=int,   default=6)
parser.add_argument("--num-heads",   type=int,   default=8)
parser.add_argument("--mlp-dim",     type=int,   default=512)
parser.add_argument("--dropout",     type=float, default=0.1)
parser.add_argument("--save",        type=str,   default="best_vit.pth")
args = parser.parse_args()

# ─── デバイス選択 ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ─── データ準備（CIFAR-10） ───────────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

# MPS は pin_memory 非対応、macOS の spawn 起動では num_workers=0 が安定
use_pin_memory = DEVICE.type == "cuda"
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=use_pin_memory)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

# ─── モデル ───────────────────────────────────────────────────────────────────
model = ViT(
    image_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    embed_dim=args.embed_dim,
    depth=args.depth,
    num_heads=args.num_heads,
    mlp_dim=args.mlp_dim,
    dropout=args.dropout,
    emb_dropout=args.dropout,
).to(DEVICE)

print(f"Parameters: {model.num_parameters():,}")
print(f"Config: embed_dim={args.embed_dim}, depth={args.depth}, "
      f"heads={args.num_heads}, mlp_dim={args.mlp_dim}")
print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}\n")

# ─── 損失・最適化・スケジューラ ──────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Linear Warmup → Cosine Annealing
def lr_lambda(epoch: int) -> float:
    if epoch < args.warmup_epochs:
        # Warmup: 0 → 1 に線形増加
        return (epoch + 1) / args.warmup_epochs
    # Cosine: 1 → 0 に減衰
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ─── 学習ループ ───────────────────────────────────────────────────────────────
best_acc = 0.0
header = (
    f"{'Epoch':>6} | {'LR':>8} | "
    f"{'Train Loss':>10} {'Train Acc':>10} | "
    f"{'Test Loss':>9} {'Test Acc':>9}"
)
print(header)
print("─" * len(header))

for epoch in range(1, args.epochs + 1):
    t0 = time.time()

    # ── Train ────────────────────────────────────────────────────────────
    model.train()
    train_loss_sum, train_correct = 0.0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient Clipping（学習安定化）
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss_sum += loss.item() * imgs.size(0)
        train_correct  += (logits.argmax(1) == labels).sum().item()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    train_loss = train_loss_sum / len(train_ds)
    train_acc  = train_correct  / len(train_ds) * 100

    # ── Eval ─────────────────────────────────────────────────────────────
    model.eval()
    test_loss_sum, test_correct = 0.0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            test_loss_sum += criterion(logits, labels).item() * imgs.size(0)
            test_correct  += (logits.argmax(1) == labels).sum().item()

    test_loss = test_loss_sum / len(test_ds)
    test_acc  = test_correct  / len(test_ds) * 100

    # ── ログ出力 ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    best_mark = ""
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), args.save)
        best_mark = " ◀ best"

    print(
        f"{epoch:>6}/{args.epochs} | {current_lr:>8.2e} | "
        f"{train_loss:>10.4f} {train_acc:>9.2f}% | "
        f"{test_loss:>9.4f} {test_acc:>8.2f}%"
        f"  [{elapsed:4.0f}s]{best_mark}"
    )

print(f"\n{'='*60}")
print(f"Best Test Accuracy: {best_acc:.2f}%")
print(f"Saved to: {args.save}")
