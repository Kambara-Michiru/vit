"""
LayerNorm スクラッチ実装の検証スクリプト

以下の 4 点を確認する:
    1. 出力が nn.LayerNorm と数値的にほぼ一致すること (atol=1e-5)
    2. 正規化後の平均 ≈ 0、分散 ≈ 1 であること
    3. weight=1/bias=0 のとき affine 変換が恒等であること
    4. backward が正しく通ること（勾配が NaN / Inf でないこと）
"""

import torch
import torch.nn as nn

from vit.layer_norm import LayerNorm

torch.manual_seed(42)

# ─── テスト用テンソル（ViT の典型的な中間テンソルと同形状）────────────────
#   B=4, N=65 (64 patches + 1 CLS), D=256
x = torch.randn(4, 65, 256)

print("=" * 60)
print("LayerNorm スクラッチ実装  検証レポート")
print("=" * 60)

# ────────────────────────────────────────────────────────────────────────
# Test 1: nn.LayerNorm との数値一致
# ────────────────────────────────────────────────────────────────────────
print("\n【Test 1】nn.LayerNorm との数値一致")

ref_ln    = nn.LayerNorm(256)
custom_ln = LayerNorm(256)

# 同じ weight/bias をセット（公平な比較のため）
with torch.no_grad():
    custom_ln.weight.copy_(ref_ln.weight)
    custom_ln.bias.copy_(ref_ln.bias)

ref_out    = ref_ln(x)
custom_out = custom_ln(x)

max_diff = (ref_out - custom_out).abs().max().item()
print(f"  max |custom - nn.LayerNorm| = {max_diff:.2e}")
assert max_diff < 1e-5, f"数値不一致！ max_diff={max_diff}"
print("  ✅ PASS  (atol < 1e-5)")

# ────────────────────────────────────────────────────────────────────────
# Test 2: 正規化後の統計量確認（affine なし）
# ────────────────────────────────────────────────────────────────────────
print("\n【Test 2】正規化後の平均 ≈ 0、分散 ≈ 1")

ln_no_affine = LayerNorm(256, elementwise_affine=False)
out_no_affine = ln_no_affine(x)

# 各トークンごとに最終次元 D=256 の統計量を計算
mean_per_token = out_no_affine.mean(dim=-1)   # (B, N)
var_per_token  = out_no_affine.var(dim=-1, unbiased=False)  # (B, N)

mean_max = mean_per_token.abs().max().item()
var_max  = (var_per_token - 1.0).abs().max().item()

print(f"  max |mean|      = {mean_max:.2e}  (理想: 0)")
print(f"  max |var - 1|   = {var_max:.2e}  (理想: 0)")
assert mean_max < 1e-5, f"平均がゼロにならない: {mean_max}"
assert var_max  < 1e-3, f"分散が 1 にならない:  {var_max}"
print("  ✅ PASS")

# ────────────────────────────────────────────────────────────────────────
# Test 3: weight=1, bias=0 のとき affine は恒等
# ────────────────────────────────────────────────────────────────────────
print("\n【Test 3】weight=1 / bias=0 のとき affine は恒等")

ln_default = LayerNorm(256)   # weight=1, bias=0 で初期化済み
out_default = ln_default(x)

# elementwise_affine=False と一致するはず
diff_affine = (out_default - out_no_affine).abs().max().item()
print(f"  max |affine(γ=1,β=0) - no_affine| = {diff_affine:.2e}")
assert diff_affine < 1e-6, f"不一致: {diff_affine}"
print("  ✅ PASS")

# ────────────────────────────────────────────────────────────────────────
# Test 4: backward が正常に通る
# ────────────────────────────────────────────────────────────────────────
print("\n【Test 4】backward（勾配計算）")

x_grad = x.clone().requires_grad_(True)
ln_train = LayerNorm(256)
out = ln_train(x_grad)
loss = out.sum()
loss.backward()

grad_x      = x_grad.grad
grad_weight = ln_train.weight.grad
grad_bias   = ln_train.bias.grad

assert grad_x      is not None, "x の勾配がない"
assert grad_weight is not None, "weight の勾配がない"
assert grad_bias   is not None, "bias の勾配がない"

has_nan_inf = (
    grad_x.isnan().any()      or grad_x.isinf().any()      or
    grad_weight.isnan().any() or grad_weight.isinf().any()  or
    grad_bias.isnan().any()   or grad_bias.isinf().any()
)
assert not has_nan_inf, "勾配に NaN / Inf が含まれている"

print(f"  grad_x      : shape={tuple(grad_x.shape)}, "
      f"mean={grad_x.mean().item():.4f}, std={grad_x.std().item():.4f}")
print(f"  grad_weight : shape={tuple(grad_weight.shape)}, "
      f"sum={grad_weight.sum().item():.4f}")
print(f"  grad_bias   : shape={tuple(grad_bias.shape)}, "
      f"sum={grad_bias.sum().item():.4f}")
print("  ✅ PASS")

# ────────────────────────────────────────────────────────────────────────
# Test 5: extra_repr の確認
# ────────────────────────────────────────────────────────────────────────
print("\n【Test 5】repr")
print(f"  {LayerNorm(256)}")

# ────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All tests passed! ✅")
print("=" * 60)
