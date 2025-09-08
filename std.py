import torch

# サンプルデータを作成
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"入力テンソル: {x}\n")

# ======================================================================
# Case 1: 不偏標準偏差 (unbiased=True, デフォルト)
# 分母が n-1 となる計算
# ======================================================================
print("--- Case 1: 不偏標準偏差 (unbiased=True) ---")

# Step 1: 平均値を計算
mean_unbiased = torch.mean(x)
print(f"Step 1: 平均値 = {mean_unbiased}")

# Step 2: 各要素と平均値の差（偏差）を計算
dev = x - mean_unbiased
print(f"Step 2: 偏差 = {dev}")

# Step 3: 偏差を2乗
dev_sq = dev ** 2
# または torch.pow(dev, 2)
print(f"Step 3: 偏差の2乗 = {dev_sq}")

# Step 4: 偏差の2乗の和を計算し、n-1で割る（不偏分散）
n = x.numel() # 要素数を取得
variance_unbiased = torch.sum(dev_sq) / (n - 1)
print(f"Step 4: 不偏分散 (n={n}, n-1={n-1}で割る) = {variance_unbiased}")

# Step 5: 平方根を取る
manual_std_unbiased = torch.sqrt(variance_unbiased)
print(f"Step 5: 手計算での標準偏差 = {manual_std_unbiased}\n")

# PyTorchの組み込み関数と比較
pytorch_std_unbiased = torch.std(x, unbiased=True) # unbiased=Trueはデフォルトなので省略可
print(f"torch.std(x, unbiased=True) の結果: {pytorch_std_unbiased}")
print(f"一致するか？: {torch.allclose(manual_std_unbiased, pytorch_std_unbiased)}\n")


# ======================================================================
# Case 2: 標本標準偏差 (unbiased=False)
# 分母が n となる計算
# ======================================================================
print("--- Case 2: 標本標準偏差 (unbiased=False) ---")

# Step 1, 2, 3は同じなので省略

# Step 4: 偏差の2乗の平均を計算（分散）
# torch.mean(dev_sq) と同じ
variance_biased = torch.sum(dev_sq) / n
print(f"Step 4: 標本分散 (n={n}で割る) = {variance_biased}")

# Step 5: 平方根を取る
manual_std_biased = torch.sqrt(variance_biased)
print(f"Step 5: 手計算での標準偏差 = {manual_std_biased}\n")

# PyTorchの組み込み関数と比較
pytorch_std_biased = torch.std(x, unbiased=False)
print(f"torch.std(x, unbiased=False) の結果: {pytorch_std_biased}")
print(f"一致するか？: {torch.allclose(manual_std_biased, pytorch_std_biased)}")

### 2. 特定の次元(`dim`)に沿った実装

#`torch.std()`では、`dim`引数を使って特定の次元に沿って標準偏差を計算することがよくあります。これも分解してみましょう。

#`dim`を指定した場合、ブロードキャストをうまく利用するために`keepdim=True`を指定して平均を計算するのがポイントです。

#```python
import torch

# 2次元のサンプルデータを作成
x_2d = torch.arange(1, 13, dtype=torch.float32).view(3, 4)
x_2d = torch.ones([3,4])
print(f"入力テンソル:\n{x_2d}\n")

# ======================================================================
# dim=0 (各列) に沿って計算
# ======================================================================
print("--- dim=0 (各列ごと) の標準偏差 ---")
dim_to_calc = 0

# Step 1: 平均値を計算 (keepdim=Trueが重要)
# (3, 3)のテンソルから(1, 3)のテンソルが生成され、ブロードキャスト可能になる
mean_dim0 = torch.mean(x_2d, dim=dim_to_calc, keepdim=True)
print(f"Step 1: 平均値 (keepdim=True):\n{mean_dim0}")

# Step 2: 偏差を計算
dev_dim0 = x_2d - mean_dim0
print(f"Step 2: 偏差:\n{dev_dim0}")

# Step 3: 偏差を2乗
dev_sq_dim0 = dev_dim0 ** 2
print(f"Step 3: 偏差の2乗:\n{dev_sq_dim0}")

# Step 4: 不偏分散を計算
n_dim0 = x_2d.shape[dim_to_calc] # この次元の要素数
variance_dim0 = torch.sum(dev_sq_dim0, dim=dim_to_calc) / (n_dim0 - 1)
print(f"Step 4: 不偏分散 (n={n_dim0}):\n{variance_dim0}")

# Step 5: 平方根を取る
manual_std_dim0 = torch.sqrt(variance_dim0)
print(f"Step 5: 手計算での標準偏差:\n{manual_std_dim0}\n")

# PyTorchの組み込み関数と比較
pytorch_std_dim0 = torch.std(x_2d, dim=dim_to_calc, unbiased=True)
print(f"torch.std(x, dim=0) の結果:\n{pytorch_std_dim0}")
print(f"一致するか？: {torch.allclose(manual_std_dim0, pytorch_std_dim0)}\n")

# ======================================================================
# dim=1 (各行) に沿って計算
# ======================================================================
print("--- dim=1 (各行ごと) の標準偏差 ---")
dim_to_calc = 1

# Step 1: 平均値を計算 (keepdim=True)
# (3, 3)のテンソルから(3, 1)のテンソルが生成される
mean_dim1 = torch.mean(x_2d, dim=dim_to_calc, keepdim=True)
print(f"Step 1: 平均値 (keepdim=True):\n{mean_dim1}")

# Step 2-5 は dim=0 と同様のロジック
dev_dim1 = x_2d - mean_dim1
dev_sq_dim1 = dev_dim1 ** 2
n_dim1 = x_2d.shape[dim_to_calc]
variance_dim1 = torch.sum(dev_sq_dim1, dim=dim_to_calc) / (n_dim1 - 1)
manual_std_dim1 = torch.sqrt(variance_dim1)
print(f"手計算での標準偏差:\n{manual_std_dim1}\n")

# PyTorchの組み込み関数と比較
pytorch_std_dim1 = torch.std(x_2d, dim=dim_to_calc, unbiased=True)
print(f"torch.std(x, dim=1) の結果:\n{pytorch_std_dim1}")
print(f"一致するか？: {torch.allclose(manual_std_dim1, pytorch_std_dim1)}")
##```

### まとめ

##`torch.std(x)`の内部計算は、以下のPyTorchコードで再現できます。

##- **`torch.std(x, unbiased=True)` (デフォルト)** の分解:
##  ```python
mean = torch.mean(x)
n = x.numel()
variance = torch.sum((x - mean) ** 2) / (n - 1)
std = torch.sqrt(variance)
##  ```

##- **`torch.std(x, unbiased=False)`** の分解:
##  ```python
mean = torch.mean(x)
n = x.numel()
variance = torch.sum((x - mean) ** 2) / n
# または variance = torch.mean((x - mean) ** 2)
std = torch.sqrt(variance)