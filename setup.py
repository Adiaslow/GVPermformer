from setuptools import setup, find_packages

setup(
    name="gvpermformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "pytorch-lightning>=2.0.0",
        "torch-geometric>=2.3.0",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    ],
    python_requires=">=3.8",
)
