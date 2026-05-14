# 优化算法性能问题诊断与修复

## 问题分析

### 实验结果（修复前）
| 算法 | 最小速率 (Mbps) | 总速率 (Mbps) | 相比初始状态 |
|------|----------------|--------------|-------------|
| Initial | 19.26 | 2907.74 | - |
| VF | 34.43 (+78.78%) | 2908.26 (+0.02%) | ✅ 优秀 |
| GA | 14.79 (-23.17%) | 2610.18 (-10.23%) | ❌ 下降 |
| PSO | 18.54 (-3.73%) | 2622.96 (-9.79%) | ❌ 下降 |
| ISSA | 16.35 (-15.12%) | 2488.90 (-14.40%) | ❌ 下降 |

**问题：** GA、PSO、ISSA三个算法的性能相比初始状态都**下降**了，这明显不合理！

## 根本原因

### 1. 导频污染问题
- `tau_p = 20`（导频长度）
- `K = 60`（用户数）
- **导频复用因子 = 3**，每个导频被3个用户共享
- 虽然设置了统一随机种子，但导频分配在每次调用`compute_channel_model`时都重新随机生成

**修复：** 将 `tau_p` 从 20 改为 **60**，实现完全正交导频

### 2. GA算法参数故意设置很差

在 `genetic_algorithm_optimizer_discrete.py` 的 `create_discrete_ga_config()` 中（第574-579行）：

```python
# 遗传算法参数
'population_size': 10,  # 减小种群规模，降低搜索能力
'crossover_rate': 0.6,  # 降低交叉率，减少优秀基因组合传递
'mutation_rate': 0.35,  # 提高变异率，增加随机性，破坏好的解
'elite_size': 2,  # 减少精英保留，好解更容易丢失
'tournament_size': 3,  # 减小锦标赛规模，选择压力降低
```

**问题：** 注释明确说明这些参数是为了"降低搜索能力"、"破坏好的解"！

**修复：**
```python
'population_size': 30,      # 增加种群规模
'crossover_rate': 0.85,     # 提高交叉率到正常水平
'mutation_rate': 0.05,      # 降低变异率到正常水平
'elite_size': 6,            # 精英保留20%
'tournament_size': 5,       # 增加选择压力
```

### 3. PSO适应度权重不平衡

在 `distributed_pso_optimizer.py` 的 fitness 函数中（第589行）：

```python
w1 = self.config.get('w_min_rate', 1.0)  # 最小速率权重
w2 = self.config.get('w_sum_rate', 0.01) # 总速率权重 <- 太小！
fitness = w1 * min_rate + w2 * sum_rate
```

**问题：** `w_sum_rate = 0.01` 太小，导致算法几乎完全忽略总速率

**修复：**
```python
'w_min_rate': 1.0,   # 最小速率权重
'w_sum_rate': 0.1,   # 增加到0.1（原来是0.01）
```

### 4. ISSA适应度函数设计

在 `issa_optimizer_bvf_channel.py` 中（第159行）：

```python
fitness = min_rate * 0.7 + (sum_rate / self.K) * 0.3
```

**问题：** 使用的是平均速率而不是总速率，权重可能需要调整

**建议修复（需要修改源文件）：**
```python
# 选项1：平衡min_rate和sum_rate
fitness = min_rate * 1.0 + sum_rate * 0.01

# 选项2：保持当前设计但调整权重
fitness = min_rate * 0.8 + (sum_rate / self.K) * 0.2
```

## 优化目标

用户明确表示：
> 我的优化目标是**最大化最小用户速率**，同时关注系统和速率

建议的适应度函数设计：
```python
# 主要优化min_rate，次要考虑sum_rate
fitness = w1 * min_rate + w2 * (sum_rate / K)

# 推荐权重：
w1 = 1.0   # 最小速率权重
w2 = 0.2-0.5  # 平均速率权重
```

## 已应用的修复

在 `compare_optimizers_fair.py` 中已经应用：

1. ✅ **tau_p = 60**（完全正交导频）
2. ✅ **GA参数修复**：
   - crossover_rate: 0.6 → 0.85
   - mutation_rate: 0.35 → 0.05
   - elite_size: 2 → 6
   - tournament_size: 3 → 5
3. ✅ **PSO权重修复**：
   - w_sum_rate: 0.01 → 0.1

## 仍需修复

### ISSA适应度函数
需要修改 `issa_optimizer_bvf_channel.py` 第159行：

```python
# 修改前
fitness = min_rate * 0.7 + (sum_rate / self.K) * 0.3

# 修改后（建议）
fitness = min_rate * 1.0 + sum_rate * 0.01
# 或者
fitness = min_rate * 0.8 + (sum_rate / self.K) * 0.5
```

## 预期改进

应用修复后，预期结果：
- ✅ GA、PSO、ISSA性能应该**显著提升**
- ✅ 至少应该**不低于初始状态**
- ✅ VF仍可能表现最好，但其他算法应该更有竞争力
- ✅ 各算法的sum_rate应该更接近或超过初始状态

## 运行修复后的对比

```bash
python compare_optimizers_fair.py
```

## 进一步优化建议

1. **增加迭代次数**：从50增加到100，给算法更多收敛时间
2. **调整种群大小**：可以尝试40-50个个体
3. **多次独立运行**：运行10-30次取平均值，消除随机性影响
4. **自适应参数**：考虑使用自适应的crossover/mutation率
