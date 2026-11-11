# LCP 搬家指南：套用到新问题

## 你需要准备的三样东西

不管你用什么模型（NN / XGBoost / 线性回归），LCP 只需要你提供：

| # | 输入 | 形状 | 说明 |
|---|------|------|------|
| 1 | **校准集残差** `r_cal` | `(n_cal,)` | `|y_true - y_pred|` on calibration set |
| 2 | **校准集距离矩阵** `D_cal` | `(n_cal, n_cal)` | 校准样本两两之间的距离 |
| 3 | **测试→校准距离矩阵** `D_test_cal` | `(n_test, n_cal)` | 每个测试样本到每个校准样本的距离 |

只要有了这三样，核心 LCP 就能跑，不需要 torch。

---

## 最简示例（复制即用）

```python
import numpy as np
from lcp.core import id_low_search, lcp_construction_path, lcp_alpha

# ============================================================
# 第一部分：你的模型（随便换）
# ============================================================
# 这里用一个烂模型做示范，你换成你自己的 XGBoost / NN / ...

def your_model_predict(x):
    """你的预测模型 — 你可以换成任何东西。"""
    return x[:, 0] * 0.5   # 瞎写的，替换掉

# ============================================================
# 第二部分：准备数据
# ============================================================
# x_train, y_train: 训练你的模型用的
# x_cal,  y_cal:   校准集（LCP 用）
# x_test, y_test:  测试集（要输出预测区间的）

# ... 载入你的数据 ...

# 训练模型 & 预测
yhat_cal = your_model_predict(x_cal)
yhat_test = your_model_predict(x_test)

# 计算残差（LCP 的核心输入 #1）
r_cal = np.abs(y_cal.ravel() - yhat_cal.ravel())

# ============================================================
# 第三部分：构建距离矩阵（LCP 的核心输入 #2、#3）
# ============================================================
# 关键：距离度量要选对。一般用特征空间的欧氏距离。

from scipy.spatial.distance import cdist

# 标准化特征（重要！否则量纲大的特征主导距离）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_all_scaled = scaler.fit_transform(np.vstack([x_cal, x_test]))
x_cal_scaled = x_all_scaled[:len(x_cal)]
x_test_scaled = x_all_scaled[len(x_cal):]

# 距离矩阵
D_cal = cdist(x_cal_scaled, x_cal_scaled, metric='euclidean')
D_test_cal = cdist(x_test_scaled, x_cal_scaled, metric='euclidean')

# ============================================================
# 第四部分：跑 LCP
# ============================================================
alpha = 0.05          # 95% 置信水平
h = np.median(D_cal)  # 带宽：用中位数距离做初始值，或 auto-tune

# 1. 排序校准残差
order = np.argsort(r_cal)
r_sorted = r_cal[order]
D_cal_sorted = D_cal[np.ix_(order, order)]
D_test_sorted = D_test_cal[:, order]

# 2. 核化 → 权重矩阵
H_cal = np.exp(-D_cal_sorted / h)
H_new = np.exp(-D_test_sorted / h)

# 3. 行累计和
Q = np.cumsum(H_cal, axis=1)
n_cal = len(r_cal)

# 4. 构建 V（排序残差 + Inf 哨兵）
V = np.append(r_sorted, np.inf)

# 5. 核心算法
id_low = id_low_search(V)
q_low = np.array([Q[i, id_low[i]] if id_low[i] >= 0 else 0.0 for i in range(n_cal)])
qn = Q[:, n_cal - 1]  # 注意：最后一列，Python 0-indexed

result = lcp_construction_path(
    alpha=alpha, V=V, id_low=id_low,
    q_low=q_low, qn=qn,
    Hnew=H_new, HnewT=H_new.T,
)

# 6. 每个测试样本的预测区间半宽
delta = result['deltaLCP']        # shape (n_test,)
lower = yhat_test.ravel() - delta
upper = yhat_test.ravel() + delta

print(f"覆盖率: {np.mean((y_test.ravel() >= lower) & (y_test.ravel() <= upper)):.3f}")
print(f"平均区间长度: {np.mean(upper - lower):.4f}")
```

---

## FAQ：搬家时最常踩的坑

### 1. 距离矩阵的尺度
**症状**：LCP 区间全部 = Inf，或者全部一样长。
**原因**：特征没标准化，或者带宽 h 不合适。
**修法**：
```python
from sklearn.preprocessing import StandardScaler
x_scaled = StandardScaler().fit_transform(x)
```
然后用缩放后的特征算距离。

还可以 auto-tune h：
```python
from lcp.core import auto_tune_distance

hs = np.logspace(-2, 2, 20) * np.median(D_cal)
ret = auto_tune_distance(V=r_cal, n=n_cal, hs=hs, D=D_cal, alpha=0.05, delta=0.025, B=5)
h_best = ret['h']
```

### 2. 样本量太小
**症状**：覆盖率远低于 0.95。
**原因**：n_cal < 100 时，标准 conformal 也救不了。
**修法**：校准集至少 200+ 样本，越大越好。

### 3. 残差定义
**症状**：区间不覆盖真实值。
**原因**：校准残差和测试残差必须用**同一口径**计算。
**修法**：
```python
# ✓ 正确：都用绝对值残差
r_cal = np.abs(y_cal - yhat_cal)
# ✗ 错误：校准用绝对值，测试用平方
```

### 4. 模型太差 → CP/LCP 也救不了
如果 `yhat` 本身和 `y_true` 相关性 < 0.5，先优化模型。LCP 只能让"合理的模型"给出更好的区间，不能修复烂模型。

---

## 如果你要用 NN + 完整 pipeline

需要装 torch，然后这样用：

```python
from lcp.pipeline import lcp_compare

# 需要把数据拆成 train / cal / test 三份
result = lcp_compare(
    xtrain=x_train, ytrain=y_train,
    xcalibration=x_cal, ycalibration=y_cal,
    xtest=x_test, ytest=y_test,
    alpha=0.05,
    nfolds=3,
    epochs=50,
    include_qr_variants=False,  # 只跑 CR/LCR/CLR/LCLR，快很多
)
print(result['coverages'])
print(result['lens'])
```
