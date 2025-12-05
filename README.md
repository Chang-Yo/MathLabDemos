# 用于研究一维混合高斯分布与中心极限定理的收敛性的Python数值模拟程序
本程序通过Python数值模拟来完成`Problem.md`中的探究性问题。主要是两个方面的研究：
1. 一维混合高斯分布中的**衍射峰**现象
2. 中心极限定理的收敛性规律

# 快速开始
请确保你的环境中安装了Python并且>=3.11版本。

这里推荐使用UV来进行包管理，请前往[UV官网](https://docs.astral.sh/uv/guides/install-python/)安装

将项目克隆后，运行以下命令完成环境的安装:
```bash
uv sync
```
uv安装好对应的依赖后，可以将编辑器的Python解释器指定为项目虚拟环境中的Python解释器

你可以通过`source .venv/bin/activate`来激活虚拟环境并运行脚本，也可以通过uv直接运行:
```bash
uv run Mathlab.py
```
稍后，你可以在`/img/`中看到图像输出

# 如何自定义参数
在`MathLab.py`中的`main_task1()`和`main_task2()`中都存在一个数组`parameter_sets`，你可以直接修改里面的参数数组来进行自定义。项目初始已经分别设定了12组和3组参数。

联系作者：
邮箱：[running_stream@sjtu.edu.cn](mailto:running_stream@sjtu.edu.cn)
个人主页：[ChangYo's Blog](https://chang-yo.github.io)