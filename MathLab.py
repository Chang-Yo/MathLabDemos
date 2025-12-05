import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 设置中英文字体列表 (Matplotlib 会尝试列表中的第一个，失败则尝试下一个)

# 英文/数学字体：设置 Times New Roman 为衬线字体系列的首选
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"  # 使用 Times New Roman 兼容的数学字体

# 中文字体：使用微软雅黑（Windows常用）或文泉驿（Linux常用）作为无衬线字体系列的首选
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Arial Unicode MS",
]

# 2. 指定使用哪个字体系列 (这里使用 'sans-serif' 来承载中文字体)
plt.rcParams["font.family"] = "sans-serif"

# 3. 解决负号显示问题 (如果中文字体处理负号有问题，使用英文符号)
plt.rcParams["axes.unicode_minus"] = False

# --- 配置 ---
# 随机数种子，用于确保结果可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 生成的随机数总量
NUM_SAMPLES = 6000
# 输出文件夹
OUTPUT_DIR = "img"
# ---


def create_output_directory(dir_path: str):
    """
    创建输出文件夹，如果它不存在。
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created output directory: {dir_path}")


def generate_mixed_gaussian(
    mu1: float, sigma1: float, mu2: float, sigma2: float, p: float, n_samples: int
) -> np.ndarray:
    """
    根据参数生成混合高斯分布 Z 的随机数。
    Z = X + eta * Y，其中 eta ~ Bernoulli(p)
    """
    # 1. 生成 Bernoulli 随机变量 eta
    # eta = 1 的概率为 p，eta = 0 的概率为 1-p
    eta = np.random.choice([1, 0], size=n_samples, p=[p, 1 - p])

    # 2. 生成 X 和 Y 的随机数
    # 注意：np.random.normal 的第二个参数是标准差 sigma
    X = np.random.normal(mu1, sigma1, size=n_samples)
    Y = np.random.normal(mu2, sigma2, size=n_samples)

    # 3. 计算 Z = X + eta * Y
    Z = X + eta * Y

    return Z


def calculate_ez_dz(
    mu1: float, sigma1: float, mu2: float, sigma2: float, p: float
) -> tuple[float, float]:
    """
    计算混合高斯分布 Z 的理论期望 E[Z] 和理论方差 D[Z]。
    """
    # 期望 E[Z] = E[X + eta*Y] = E[X] + E[eta]*E[Y] = mu1 + p * mu2
    EZ = mu1 + p * mu2

    # 方差 D[Z] = D[X + eta*Y]
    # D[Z] = D[X] + D[eta*Y]
    # D[eta*Y] = E[(eta*Y)^2] - (E[eta*Y])^2
    # E[eta*Y] = p * mu2
    # E[(eta*Y)^2] = E[eta^2] * E[Y^2] = p * (sigma2^2 + mu2^2)
    # D[Z] = sigma1^2 + p * (sigma2^2 + mu2^2) - (p * mu2)^2
    DZ = sigma1**2 + p * sigma2**2 + p * mu2**2 * (1 - p)

    return EZ, DZ


def generate_u_distro(
    mu1: float, sigma1: float, mu2: float, sigma2: float, p: float, n: int
) -> np.ndarray:
    """
    生成 1000 个标准化样本和 U 的值。
    U = ( sum(Z_j) - n * E[Z] ) / sqrt(n * D[Z])

    Args:
        mu1, sigma1, mu2, sigma2, p: 混合高斯分布的参数。
        n: 每组样本的数量。

    Returns:
        np.ndarray: 包含 1000 个 U 值的数组。
    """
    NUM_GROUPS = 1000  # 任务要求的 U 值数量

    # 1. 计算理论期望和方差
    EZ, DZ = calculate_ez_dz(mu1, sigma1, mu2, sigma2, p)

    # 确保方差大于 0，避免除以零
    if DZ <= 1e-9:
        print("Warning: Theoretical variance D[Z] is near zero. Skipping this case.")
        return np.array([])

    # 2. 生成 1000 组，每组 n 个 Z 样本
    # Z_samples shape: (NUM_GROUPS * n)
    Z_samples_flat = generate_mixed_gaussian(
        mu1, sigma1, mu2, sigma2, p, NUM_GROUPS * n
    )

    # Reshape 成 (NUM_GROUPS, n)
    Z_samples = Z_samples_flat.reshape(NUM_GROUPS, n)

    # 3. 计算每组的样本和 Sum_Z = sum(Z_j)
    Sum_Z = np.sum(Z_samples, axis=1)  # shape: (1000,)

    # 4. 计算 U 值
    # U = ( Sum_Z - n * E[Z] ) / sqrt(n * D[Z])
    U = (Sum_Z - n * EZ) / np.sqrt(n * DZ)

    return U


def plot_histogram(
    data: np.ndarray,
    output_dir: str,
    # Task 1 Parameters
    mu1: float = None,
    sigma1: float = None,
    mu2: float = None,
    sigma2: float = None,
    p: float = None,
    # Task 2 Parameters
    n: int = None,
):
    """
    绘制随机数数据的频率分布直方图，并保存图像。
    此函数现在支持任务 1 (Z 的分布) 和任务 2 (U 的分布)。
    """
    is_task2 = n is not None

    plt.figure(figsize=(10, 6))

    # 绘制频率直方图
    n_bins = int(np.sqrt(len(data)))
    plt.hist(
        data,
        bins=n_bins,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Frequency Distribution",
    )

    if is_task2:
        # --- 任务 2: U 的分布，应对比标准正态分布 ---
        filename = f"FrequencyDistro-mu_{mu2}-n_{n}.png"

        # 绘制标准正态分布的 PDF
        x = np.linspace(min(data), max(data), 100)
        # 标准正态分布 PDF: f(x) = 1/sqrt(2*pi) * exp(-x^2/2)
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
        plt.plot(x, pdf, "r-", linewidth=2, label="Standard Normal PDF $N(0, 1)$")

        plt.title(
            f"Distribution of Standardized Sample Sum $U$ (n={n}, Groups=1000)",
            fontsize=14,
        )
        plt.xlabel(f"Value of $U$ (Approximation to $N(0, 1)$)", fontsize=12)

        # 标注所使用的参数集
        params_text = (
            f"$\mu_1$={mu1}, $\sigma_1$={sigma1}\n"
            f"$\mu_2$={mu2}, $\sigma_2$={sigma2}\n"
            f"$p$={p}"
        )
        plt.text(
            0.98,
            0.95,
            params_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6),
        )

        plt.legend(loc="upper left")

    else:
        # --- 任务 1: Z 的分布 ---
        filename = f"{mu1}-{sigma1}-{mu2}-{sigma2}-{p}.png"

        EZ, DZ = calculate_ez_dz(mu1, sigma1, mu2, sigma2, p)

        plt.title(
            f"Frequency Distribution of Mixed Gaussian $Z$ (N={len(data)})", fontsize=14
        )
        plt.xlabel(f"Value of $Z$", fontsize=12)

        # 标注参数信息
        params_text = (
            f"$\mu_1$={mu1}, $\sigma_1$={sigma1}\n"
            f"$\mu_2$={mu2}, $\sigma_2$={sigma2}\n"
            f"$p$={p}\n"
            f"$E[Z]\\approx{EZ:.2f}$\n"
            f"$D[Z]\\approx{DZ:.2f}$"
        )
        plt.text(
            0.98,
            0.95,
            params_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6),
        )

    plt.ylabel("Frequency (Density)", fontsize=12)
    plt.grid(axis="y", alpha=0.5, linestyle="--")
    plt.tight_layout()

    # 保存图像
    filepath = os.path.join(output_dir, filename)
    DPI_SETTING = 300
    plt.savefig(filepath, dpi=DPI_SETTING)
    plt.close()  # 关闭当前图像，释放内存
    print(f"Successfully generated and saved: {filepath}")


def main_task1():
    """
    主函数：定义参数并执行生成和绘图任务。
    """
    create_output_directory(OUTPUT_DIR)

    # --- 设定不同的参数 ---
    # 您的要求是设定不同的参数，这里给出三组具有代表性的参数
    # 注意：传入函数的是标准差 (sigma)，但参数表格中使用的是方差 (sigma^2)

    # --- 扩展后的参数设定（共 8 组） ---
    parameter_sets = [
        # 控制mu_1增大
        {"mu1": 1, "sigma1": 1, "mu2": 0, "sigma2": 1, "p": 0.5},
        {"mu1": 2, "sigma1": 1, "mu2": 0, "sigma2": 1, "p": 0.5},
        {"mu1": 4, "sigma1": 1, "mu2": 0, "sigma2": 1, "p": 0.5},
        # 控制mu_2增大
        {"mu1": 0, "sigma1": 1, "mu2": 0, "sigma2": 1, "p": 0.5},
        {"mu1": 0, "sigma1": 1, "mu2": 2, "sigma2": 1, "p": 0.5},
        {"mu1": 0, "sigma1": 1, "mu2": 4, "sigma2": 1, "p": 0.5},
        # 减小 sigma_1 和 sigma_2
        {"mu1": 0, "sigma1": 1, "mu2": 4, "sigma2": 0.5, "p": 0.5},
        {"mu1": 0, "sigma1": 0.5, "mu2": 4, "sigma2": 1, "p": 0.5},
        {"mu1": 0, "sigma1": 0.5, "mu2": 4, "sigma2": 0.5, "p": 0.5},
        # 改变 p
        {"mu1": 0, "sigma1": 1, "mu2": 5, "sigma2": 1, "p": 0.2},
        {"mu1": 0, "sigma1": 1, "mu2": 5, "sigma2": 1, "p": 0.5},
        {"mu1": 0, "sigma1": 1, "mu2": 5, "sigma2": 1, "p": 0.7},
    ]

    print("--- Starting Task 1: Mixed Gaussian Generation and Plotting ---")

    for params in parameter_sets:
        mu1 = params["mu1"]
        sigma1 = params["sigma1"]
        mu2 = params["mu2"]
        sigma2 = params["sigma2"]
        p = params["p"]

        print(
            f"Processing parameters: mu1={mu1}, sigma1^2={sigma1**2}, mu2={mu2}, sigma2^2={sigma2**2}, p={p}"
        )

        # 1. 生成混合高斯分布随机数
        Z = generate_mixed_gaussian(mu1, sigma1, mu2, sigma2, p, NUM_SAMPLES)

        # 2. 绘制频率分布直方图并保存
        # 注意：文件名要求使用 sigma_1 和 sigma_2 的具体数值 (即标准差)
        plot_histogram(Z, OUTPUT_DIR, mu1, sigma1, mu2, sigma2, p)

    print("--- Task 1 Completed ---")


def main_task2():
    """
    主函数：执行任务 2，验证中心极限定理。
    """
    create_output_directory(OUTPUT_DIR)

    # --- 选择用于任务 2 的一组参数 ---
    # 通常选用一个能体现非正态分布特征的参数集，我们使用第一组：
    # 理论上，此参数集 Z 的分布是双峰或偏斜的 (取决于 mu1, mu2 差异和 p)
    parameter_sets = [
        {"mu1": 0, "sigma1": 1, "mu2": 2, "sigma2": 1, "p": 0.5},
        {"mu1": 0, "sigma1": 1, "mu2": 5, "sigma2": 1, "p": 0.5},
        {"mu1": 0, "sigma1": 1, "mu2": 10, "sigma2": 1, "p": 0.5},
    ]
    # params = {"mu1": 0, "sigma1": 1, "mu2": 10, "sigma2": 1, "p": 0.5}

    print("\n--- Starting Task 2: Central Limit Theorem Verification ---")

    for params in parameter_sets:
        mu1 = params["mu1"]
        sigma1 = params["sigma1"]
        mu2 = params["mu2"]
        sigma2 = params["sigma2"]
        p = params["p"]

        EZ, DZ = calculate_ez_dz(mu1, sigma1, mu2, sigma2, p)

        print(f"Base Z Parameters: E[Z]={EZ:.4f}, D[Z]={DZ:.4f}")

        # --- n 的取值要求 ---
        n_values = [2, 3, 4, 5, 10, 20, 50, 100, 5000]

        for n in n_values:
            print(f"Processing n={n}...")

            # 1. 生成 1000 个 U 值
            U_distro = generate_u_distro(mu1, sigma1, mu2, sigma2, p, n)

            # 只有在成功生成数据时才绘图
            if len(U_distro) > 0:
                # 2. 绘制频率分布直方图并保存
                # 传入 Task 2 特定的参数 n，以及 Task 1 的参数用于标注
                plot_histogram(
                    U_distro,
                    OUTPUT_DIR,
                    mu1=mu1,
                    sigma1=sigma1,
                    mu2=mu2,
                    sigma2=sigma2,
                    p=p,
                    n=n,
                )

    print("--- Task 2 Completed ---")


if __name__ == "__main__":
    # main_task1()
    main_task2()
