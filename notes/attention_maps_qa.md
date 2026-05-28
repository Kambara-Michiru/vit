# Attention Map 可視化 Q&A

---

## Q1. ヘッド別アテンションマップとは何を見ているのか？

**A.** 1つの Transformer 層の中にある「8個の独立した注意の視点」を、ヘッドごとに並べたものです。

Multi-Head Attention では、入力 `(B, 65, 256)` を 8 分割して、それぞれ別の部分空間で注意を計算します。

```
ヘッド0 → (B, 65, 32) で注意計算  → patch_3 と patch_51 に集中
ヘッド1 → (B, 65, 32) で注意計算  → patch_12〜20 のまとまりに集中
...
ヘッド7 → (B, 65, 32) で注意計算  → 全体に均一に注目
```

アテンション重みの形状は `(B, H=8, N=65, N=65)` で、`[b, h, i, j]` の値は「トークン i がトークン j にどれだけ注目しているか」を表します（softmax 後なので行の合計 = 1）。

---

## Q2. ViT には 6 層あるのに、ヘッド別マップはどの層を見ているのか？

**A.** **最終層（第 6 層）のみ**です。

全層の重みを `forward hook` で取得しており、ヘッド別マップにはその中の最終層だけを使っています。

```python
# block.attn.attn_drop に hook を差し込む
attn_maps = {}
for i, block in enumerate(model.transformer.layers):
    block.attn.attn_drop.register_forward_hook(
        lambda m, inp, out, idx=i: attn_maps.__setitem__(idx, out.detach().cpu())
    )

model(img_batch)          # forward → 全6層が attn_maps に貯まる
last = attn_maps[5]       # 最終層: shape (1, 8, 65, 65)
```

最終層はクラス分類に最も直結しているため、「このヘッドが何を見ているか」を確認する目的には十分です。

---

## Q3. なぜ `attn_drop` に hook するのか？

**A.** softmax 後・dropout 後のアテンション重みを取るためです。

Attention の計算フローは次のとおりです。

```
q @ k.T * scale  →  softmax  →  attn_drop  →  @ v
```

`attn_drop` の **入力** が softmax 済みのアテンション重みなので、そこに hook を差し込むと「実際に v に掛けられた重み」がそのまま取れます。モデルのコードを一切変更せずに済むのもメリットです。

---

## Q4. 「全層をまたがって」可視化したいときはどうするのか？

**A.** **Attention Rollout** を使います。

最終層だけ見る方法では「前の層でどう情報が流れたか」が分かりません。Rollout は各層のアテンション行列を掛け合わせることで、「層1 → 層2 → … → 層6 を経て CLS に届いた情報フロー」を1枚のマップにまとめます。

```python
# viz_utils.py: attention_rollout()
rollout = torch.eye(N)           # 恒等行列（残差接続のモデル化）

for attn in attn_per_layer:      # 6層分ループ
    attn_avg = attn.mean(dim=0)              # 8ヘッドを平均 → (N, N)
    attn_aug = attn_avg + torch.eye(N)       # 残差接続を加算
    attn_aug /= attn_aug.sum(-1, keepdim=True)  # 行正規化
    rollout  = attn_aug @ rollout            # 行列積で積算

return rollout[0, 1:]  # CLS 行のパッチ部分: (64,)
```

**なぜ恒等行列を足すのか？**  
Transformer には残差接続（`x + Attention(x)`）があるため、アテンション重みが低くても「素通りした情報」が次の層に届きます。恒等行列を加算することでこの素通りパスを近似しています（Abnar & Zuidema, 2020）。

---

## Q5. 3種類の可視化を使い分けるには？

| 可視化 | 見ているもの | 適した用途 |
|---|---|---|
| **ヘッド別マップ** | 最終層・ヘッドごとの注意パターン | ヘッドの役割分担を確認したい |
| **CLS アテンション（層別）** | 各層で CLS が注目しているパッチ | 層を深めるごとに注意がどう変わるか確認したい |
| **Attention Rollout** | 全6層を合成した情報フロー | 最終的にどのパッチが分類に効いたか確認したい |

---

## 参考

- Vaswani et al. (2017) [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Dosovitskiy et al. (2021) [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Abnar & Zuidema (2020) [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)（Attention Rollout の出典）
