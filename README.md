# Vision Transformer (ViT) スクラッチ実装

論文「[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)」(Dosovitskiy et al., ICLR 2021) の PyTorch スクラッチ実装。  
CIFAR-10（32×32 画像、patch_size=4）で動作確認済み。

---

## ファイル構成

```
vit/
├── vit/
│   ├── __init__.py
│   ├── patch_embedding.py   # ① 画像 → パッチ列
│   ├── attention.py         # ② Multi-Head Self-Attention
│   ├── transformer.py       # ③ Encoder Block (LN + Attn + LN + FFN)
│   ├── layer_norm.py        # Layer Normalization スクラッチ実装
│   └── vit.py               # ④ ViT 本体 (CLS token, Pos Embed, Head)
├── notebooks/
│   ├── 01_modules.ipynb        # モジュール形状追跡
│   ├── 02_attention_maps.ipynb # Attention 可視化
│   ├── 03_pos_embedding.ipynb  # Positional Embedding 可視化
│   ├── viz_utils.py            # 共通描画ユーティリティ
│   └── build_notebooks.py      # ノートブック生成スクリプト
├── train.py                 # CIFAR-10 学習スクリプト
├── check_model.py           # forward pass 動作確認
├── test_layer_norm.py       # LayerNorm 数値一致テスト
└── requirements.txt
```

---

## セットアップ

```bash
git clone <this-repo>
cd vit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## モデルアーキテクチャ

```
入力画像 (B, 3, 32, 32)
    ↓ ① PatchEmbedding     → (B, 64, 256)   # 4×4 パッチ × 256 次元
    ↓ ② CLS token 付加     → (B, 65, 256)   # 先頭に分類トークン
    ↓ ③ Positional Embed   → (B, 65, 256)   # 位置情報を加算
    ↓ ④ Transformer ×6     → (B, 65, 256)   # Attention + FFN
    ↓ ⑤ CLS token 取り出し → (B, 256)
    ↓ ⑥ MLP Head           → (B, 10)        # クラス確率
```

| ハイパーパラメータ | 値 |
|---|---|
| image_size | 32 |
| patch_size | 4（→ 8×8 = 64 patches）|
| embed_dim | 256 |
| depth | 6 |
| num_heads | 8（head_dim = 32）|
| mlp_dim | 512 |
| パラメータ数 | 約 319 万 |

---

## 学習

```bash
# デフォルト設定（30 エポック）
python train.py

# カスタム設定
python train.py --epochs 50 --batch-size 256 --lr 1e-3
```

**学習レシピ**

| 項目 | 設定 |
|---|---|
| Optimizer | AdamW (weight_decay=0.05) |
| LR Schedule | Linear Warmup (5 epoch) + Cosine Annealing |
| 正則化 | Label Smoothing 0.1, Dropout 0.1, Grad Clip 1.0 |
| Augmentation | RandomCrop(padding=4), RandomHorizontalFlip |
| デバイス | MPS / CUDA / CPU を自動選択 |

**CIFAR-10 テスト精度**

| エポック | Test Acc |
|---|---|
| 10 | 56.5% |
| 20 | 65.6% |
| 30 | **70.8%** |

---

## 動作確認

```bash
# forward pass の形状チェック
python check_model.py

# LayerNorm の数値一致テスト（nn.LayerNorm との比較）
python test_layer_norm.py
```

---

## インタラクティブ探索（Jupyter Notebook）

学習済みモデル（`best_vit.pth`）と CIFAR-10 データが必要。

```bash
cd notebooks
jupyter lab
```

### 📒 01_modules.ipynb — モジュール形状追跡

ViT の各モジュールをセル 1 つずつ実行し、テンソルの形状と値の変化を追う。

| セクション | 内容 | 可視化 |
|---|---|---|
| ① Patch Embedding | 32×32 → 8×8 グリッド | パッチ分割グリッド |
| ② CLS Token | シーケンス先頭に付加 | 全トークンの L2 ノルム棒グラフ |
| ③ Positional Embedding | 位置情報の加算 | 加算前後のノルム比較 ＋ 8×8 ノルムマップ |
| ④ Transformer ×6 | 各層での表現の成長 | CLS ノルム推移グラフ |
| ⑤ Classification Head | CLS → クラス確率 | Softmax 確率の水平棒グラフ |

### 🔥 02_attention_maps.ipynb — Attention 可視化

`block.attn.attn_drop` へ **forward hook** を差し込み、`(B, H, N, N)` のアテンション重みをキャプチャする。

| セクション | 内容 | 可視化 |
|---|---|---|
| Hook の登録 | `attn_drop` に hook を設定 | — |
| ヘッド別マップ | 最終層の 8 ヘッドを並べる | 65×65 アテンションマップ |
| CLS アテンション（全層） | CLS がどのパッチを見ているか | ヒートマップ重ね合わせ |
| 最終層・各ヘッド | ヘッドごとの注目箇所の違い | 8 枚のヒートマップ |
| Attention Rollout | 全 6 層を行列積で合成 | 単一の「最終注目マップ」 |

> **Attention Rollout** (Abnar & Zuidema, 2020): 各層の注意行列に恒等行列（残差接続を模倣）を加算・正規化し積算することで、入力パッチから CLS トークンへの情報フローを推定する。

### 🗺️ 03_pos_embedding.ipynb — Positional Embedding 可視化

`model.pos_embed`（形状 `(1, 65, 256)`）の内部構造を多角的に可視化する。

| セクション | 内容 | 読み取り方 |
|---|---|---|
| コサイン類似度マトリクス | 65×65 のヒートマップ | 隣接パッチが近い色 → 空間位置が学習できている |
| 各位置の類似度マップ | 1 パッチ vs 全パッチ の類似度（4 箇所） | 距離勾配が連続的かを確認 |
| PCA 2D プロット | 256 次元を 2 次元に圧縮 | 64 パッチが格子状に並ぶほど位置情報が整合 |

---

## 実装上のポイント

### LayerNorm スクラッチ実装

`nn.LayerNorm` の代わりに `vit/layer_norm.py` を使用。

```python
# 正規化の核心部分
mean  = x.mean(dim=-1, keepdim=True)
var   = x.var(dim=-1, keepdim=True, unbiased=False)   # 母分散
x_hat = (x - mean) / torch.sqrt(var + self.eps)
return self.weight * x_hat + self.bias
```

`nn.LayerNorm` との最大差は `4.77e-07`（float32 精度内）。

### Multi-Head Attention の形状変換

```python
# D を (H ヘッド × head_dim) に分解してから H を前に移動
q = q.reshape(B, N, num_heads, head_dim).transpose(1, 2)  # (B, H, N, head_dim)
#   ↑ reshape だけでは H が内側に残って @ で一括計算できない
#     transpose で H をバッチ次元に出すのが重要
attn = (q @ k.transpose(-2, -1)) * scale   # (B, H, N, N)
```

### fused QKV

Q/K/V を 3 本の Linear で個別計算するより、**1 本の Linear(D→3D)** で一括計算する方が GEMM 回数が 1/3 になり効率的。

```python
q, k, v = self.qkv(x).chunk(3, dim=-1)   # 1 回の GEMM → 均等分割
```

---

## 参考文献

- Dosovitskiy et al. (2021) [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Ba et al. (2016) [Layer Normalization](https://arxiv.org/abs/1607.06450)
- Abnar & Zuidema (2020) [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)（Attention Rollout）

## 参考 OSS

- [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
- [tintn/vision-transformer-from-scratch](https://github.com/tintn/vision-transformer-from-scratch)
- [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
