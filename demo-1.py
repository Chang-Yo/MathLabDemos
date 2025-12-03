import numpy as np
import matplotlib.pyplot as plt

# --- Matplotlib 中文支持设置 ---
# 尝试设置字体为 WenQuanYi Micro Hei（许多Linux/WSL默认自带）
# 如果这个字体不存在，可能需要安装 ttf-wqy-microhei 包

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]

plt.rcParams["axes.unicode_minus"] = False  # 解决负号 '-' 显示为方块的问题

plt.style.use("ggplot")
# 1. Define parameters
lambda_poisson = 2
sample_sizes = [10, 30, 50, 70, 100]  # n values
num_samples = 1000  # Number of samples (k=50) for each n

# Set up the figure for plotting
fig, axes = plt.subplots(
    nrows=len(sample_sizes), ncols=1, figsize=(10, 3 * len(sample_sizes))
)
plt.subplots_adjust(hspace=0.5)

# Ensure axes is an array even for a single subplot
if len(sample_sizes) == 1:
    axes = [axes]


# 2. Loop through each sample size (n)
for i, n in enumerate(sample_sizes):
    # Store the 50 sample means for the current n
    sample_means = []

    # 3. Generate 50 samples and their means
    for _ in range(num_samples):
        # Generate n random numbers from Poisson(lambda=2)
        # The loc=0, size=n is for consistency, but numpy's poisson defaults to 0
        sample = np.random.poisson(lam=lambda_poisson, size=n)

        # Calculate the sample mean
        mean = np.mean(sample)
        sample_means.append(mean)

    # Convert to a numpy array for easier calculation
    sample_means = np.array(sample_means)

    # 4. Calculate descriptive statistics for the 50 sample means
    mean_of_means = np.mean(sample_means)
    std_of_means = np.std(sample_means)
    theoretical_std_err = np.sqrt(lambda_poisson / n)

    # 5. Plot the histogram
    ax = axes[i]
    # Use 10 bins for a clear view of the distribution
    ax.hist(
        sample_means,
        bins=10,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    # Add theoretical mean (E[X]=lambda=2) and standard error lines
    ax.axvline(
        lambda_poisson,
        color="red",
        linestyle="dashed",
        linewidth=1.5,
        label=f"理论均值 $\mu={lambda_poisson}$",
    )

    # Add a title and labels
    ax.set_title(
        f"样本均值的频率分布直方图 (n={n})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("样本均值 ($\\bar{X}$)", fontsize=12)
    ax.set_ylabel("频率 (Normalized)", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.5)

    # 6. Print the statistics
    print(f"--- Analysis for Sample Size n = {n} ---")
    print(f"  Theoretical Population Mean (λ): {lambda_poisson}")
    print(f"  Mean of {num_samples} Sample Means: {mean_of_means:.4f}")
    print(f"  Standard Deviation (SD) of Sample Means: {std_of_means:.4f}")
    print(
        f"  Theoretical Standard Error (SE = $\\sqrt{{\\lambda/n}}$): {theoretical_std_err:.4f}"
    )
    print("-" * 35)

plt.show()
