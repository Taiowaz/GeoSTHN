from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "sampler_core",
        ["src/utils/sampler_core.cpp"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="anonymous-hsact",
    version="0.1.0",
    description="Anonymous implementation for temporal heterogeneous graph link prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    setup_requires=["pybind11>=2.5"],
    python_requires=">=3.9",
)
