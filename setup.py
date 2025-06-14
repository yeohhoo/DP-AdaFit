from setuptools import setup

setup(
    name="DP-AdaFit",
    py_modules=["DP-AdaFit"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
