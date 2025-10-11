# 导入必要的模块
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# --- C++ 扩展模块定义 ---
# 这部分保留了你原有的配置，用于编译 C++ 代码
ext_modules = [
    Pybind11Extension(
        # 扩展模块的名称，编译后你可以通过 `import sampler_core` 来使用
        "sampler_core",
        # 源文件列表
        ["src/utils/sampler_core.cpp"],
        # 添加编译和链接参数以支持 OpenMP 并行化
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

# --- 读取 README.md 作为长描述 ---
# 这是一个良好实践，它会把你的项目介绍放到包的主页上
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


# --- 主设置函数 ---
setup(
    # 1. 项目元数据
    name="geosthn",  # 建议使用一个能代表你整个项目的名字
    version="0.1.0",     # 项目版本号
    author="Albin",  # 你的名字或团队名 (替换 XXXX-2)
    author_email="taiowaz@gmail.com", # 你的邮箱 (替换 XXXX-3)
    url="https://github.com/Taiowaz/GeoSTHN",  # 项目主页 (替换 XXXX-4)
    long_description=long_description,
    long_description_content_type="text/markdown", # 指定长描述的格式为 Markdown

    # 2. 包发现 (关键优化)
    # find_packages() 会自动查找项目中所有的 Python 包 (即包含 __init__.py 的文件夹)
    # 这将确保 `src` 和 `tgb` 两个目录被正确识别和安装。
    packages=find_packages(),

    # 3. C++ 扩展模块
    # 告诉 setuptools 存在需要编译的 C++ 扩展
    ext_modules=ext_modules,

    # 4. 构建系统配置
    # 指定用于构建 C++ 扩展的命令类，确保 pybind11 被正确处理
    cmdclass={"build_ext": build_ext},
    # 声明构建时的依赖，setuptools 会确保在编译前 pybind11 已被安装
    setup_requires=["pybind11>=2.5"],

    # # 5. 其他信息 (可选但推荐)
    # # 声明项目运行所需的依赖库，pip 在安装此项目时会自动安装它们
    # install_requires=[
    #     "numpy",
    #     "pandas",
    #     # "torch", # 根据你的项目需要添加其他依赖
    # ],
    # 声明你的项目兼容的 Python 版本
    python_requires=">=3.9",
)