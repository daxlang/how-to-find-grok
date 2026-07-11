# Grokking Arc: 无需验证集的泛化检测

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**当神经网络发生"顿悟"——在完全背下训练集后突然开始泛化——它的隐藏表示会发生可测量的维度塌缩。我们把这个指标叫 `erank`，它的轨迹会形成可预测的弧形，弧底就是最佳停止点。**

## erank 是什么

`erank`（有效秩）测量隐藏层表示占据了多少独立方向。取第一隐藏层的激活矩阵 `H [N × d]`，做 SVD 得到奇异值 `σ₁ ≥ σ₂ ≥ ...`：

```
pᵢ = σᵢ / Σⱼ σⱼ
erank = exp(-Σᵢ pᵢ · log pᵢ)
```

- **erank ≈ hidden_dim**（高）：每个样本独立占一个方向 → 记忆态，泛化差
- **erank ≪ hidden_dim**（低）：表示压缩到少数方向 → 泛化态
- **erank 反转（140 - erank）**：越高 = 越泛化，与准确率同向

一行 PyTorch 计算：
```python
_, S, _ = torch.linalg.svd(H.float(), full_matrices=False)
erank = torch.exp(-(p := S/S.sum() * torch.log(p+1e-10)).sum()).item()
```

## 实验设置

```
任务:    (a + b) % 97   —— 模加法
模型:    2层 MLP (embed=128, hidden=256)
数据:    训练集 50%（~4700 样本），测试集全部 9409 对
噪声:    label_noise_ratio 比例的训练标签随机篡改
优化:    AdamW, lr=0.001, batch=512
变量:    weight_decay (0.0-5.0), label_noise (0%-40%), dim (32-256)
记录:    每 50 epoch 记录 train_loss, test_acc, erank
```

## 核心发现

### 1. erank 忠实追踪每一次 Grokking

![grokking](figures/fig1_grokking.png?v=3)

蓝线 = 140 - erank（越高 = 越压缩），红虚线 = 准确率。**左**：wd=0.5 下 erank 从 138→60（反转后 2→80），准确率 2%→100%，两条曲线几乎重合。**右**：wd=0.0 下 erank 纹丝不动——模型从未压缩，准确率永远 50%。

### 2. 不同噪声下的最优 Weight Decay

![best wd](figures/fig2_best_wd.png?v=3)

每个噪声等级下，展示该等级最优 wd 的 erank 轨迹和准确率：
- noise=0%：最优 wd=0.3，最终 erank=46，准确率 100%。微弧（erank 从 138→46 后缓慢升至 60）。
- noise=10%：最优 wd 升至 2.0，最终 erank=93，准确率 95%。清晰深弧。
- noise=20%：最优 wd 升至 2.15，最终 erank=93，准确率 95%。弧仍存在但窗口更窄。
- noise=40%：**无泛化**。所有 wd 都无法 grok，准确率卡在 ~30%。

**结论：噪声越高，需要越强的 wd 来压制噪声反扑。但存在上限——噪声超过 ~30% 后，没有任何 wd 能实现泛化。**

### 3. 最优 wd 随噪声增加而增大

![wd vs noise](figures/fig3_wd_vs_noise.png?v=3)

左图：可泛化的 wd 从 noise=0 时的 0.3 飙升到 noise=20% 时的 2.15。右图：峰值准确率从 100% 降至 95%，噪声=40% 时完全无法 grok。

### 4. 固定 wd=0.5 在噪声增大时迅速失效

![noise wd05](figures/fig4_noise_wd05.png?v=3)

wd=0.5 在 noise=0 时能让模型完美泛化（erank=59, acc=100%），但在 noise=10% 时 erank 停在 119（压缩不足），在 noise≥20% 时完全无法 grok。**这说明 wd 必须随噪声调整，固定的 wd 只在特定噪声范围内有效。**

## 如何使用 erank 选择 Weight Decay

训练时监控 erank 轨迹，根据曲线形状判断 wd 是否合适：

![wd regimes](figures/fig5_wd_regimes.png?v=3)

**左：wd 过小** — erank 保持高位，不压缩。**增大 wd。**

**中：wd 刚好（弧形区）** — erank 先骤降后回升，弧底 = 最优泛化。**在弧底早停。**

**右：wd 过大** — erank 被暴力压到 0，模型死亡。**减小 wd。**

### 调参流程

1. 初始 wd=0.5，观察 erank 曲线
2. 无压缩 → 翻倍（1.0, 2.0...）
3. 塌死 → 减半（0.25, 0.12...）
4. 出现弧形 → 记录弧底 epoch 为早停点

## 快速开始

```bash
pip install torch numpy matplotlib
python src/run.py --mode grok --device cuda      # Grokking 全轨迹
python src/run.py --mode arc --device cuda        # 弧形扫描
python src/run.py --mode noise --device cuda      # 噪声扫描
python src/plot.py                                # 重新生成图表
```

## 数据文件

| 文件 | 内容 |
|------|------|
| `grokking_trajectory.json` | Grokking 全追踪 (20000 epochs) |
| `arc_discovery_noise01.json` | Noise=10% wd 扫描 —— 弧形首次发现 |
| `arc_high_wd_noise01.json` | Noise=10% 高 wd (2.0/3.0/5.0) 对比 |
| `arc_window_noise02.json` | Noise=20% wd 精细扫描 |
| `arc_optimal_noise02.json` | Noise=20% wd 2.0-2.2 最优定位 |
| `arc_broad_noise02.json` | Noise=20% wd 0.1-2.0 宽扫描 |
| `arc_noise04_closed.json` | Noise=40% wd 扫描 —— 窗口关闭 |
| `noise_scan.json` | 同一 wd=0.5 在不同噪声下的表现 |

## 相关工作

用秩/维度作为 grokking 序参量的独立工作：
- Wang (2026): *Grokking as Dimensional Phase Transition*
- ERI Labs (2026): *Fisher Rank Crystallization*
- DeMoss et al. (2024): *Complexity Dynamics of Grokking*
