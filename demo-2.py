import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Matplotlib 中文支持设置 ---
# 设置字体为 WenQuanYi Micro Hei 或 SimHei
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题
# -------------------------------

# 应用样式以提高美观度
plt.style.use("ggplot")

# 1. 定义新的参数
lambda_poisson = 2
# ⚠️ 样本容量 n 仅为 1000
sample_sizes = [10000]
# ⚠️ 随机样本数量 k 增加到 1000
num_samples = 10000

# 设置绘图区域
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
plt.subplots_adjust(hspace=0.5)

# --- 主循环，只执行一次（n=1000）---
n = sample_sizes[0]
sample_means = []

# 2. 生成 k=1000 个样本和它们的均值
for _ in range(num_samples):
    # 生成 n=1000 个随机数
    sample = np.random.poisson(lam=lambda_poisson, size=n)
    mean = np.mean(sample)
    sample_means.append(mean)

# 转换为 numpy 数组
sample_means = np.array(sample_means)

# 3. 计算统计数据
mean_of_means = np.mean(sample_means)
std_of_means = np.std(sample_means)
theoretical_std_err = np.sqrt(lambda_poisson / n)

# 4. 绘制直方图
# 由于分布非常集中，我们使用更多的 bins 来细化展示
ax.hist(sample_means, bins=30, density=True, alpha=0.7, color="teal", edgecolor="black")

# 添加理论均值线
ax.axvline(
    lambda_poisson,
    color="red",
    linestyle="dashed",
    linewidth=1.5,
    label=f"理论均值 $\mu={lambda_poisson}$",
)

# 设置标题和标签（使用中文）
ax.set_title(
    f"样本均值的频率分布直方图 (n={n}, k={num_samples})", fontsize=14, fontweight="bold"
)
ax.set_xlabel("样本均值 ($\\bar{X}$)", fontsize=12)
ax.set_ylabel("频率 (Normalized)", fontsize=12)
ax.legend()
ax.grid(axis="y", alpha=0.5)

# 5. 打印统计信息
print(f"--- 最终分析 (n={n}, k={num_samples}) ---")
print(f"  理论总体均值 (λ): {lambda_poisson}")
print(f"  1000 个样本均值的平均值: {mean_of_means:.6f}")
print(f"  样本均值的标准差 (实验标准误差): {std_of_means:.6f}")
print(f"  理论标准误差 (SE = $\\sqrt{{\\lambda/n}}$): {theoretical_std_err:.6f}")
print("-" * 50)

plt.show()
