from setuptools import setup, find_packages
from pathlib import Path

root_dir = Path(__file__).parent
long_description = (root_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="DexPoint",
    version="0.4.0",
    author="Yuzhe Qin",
    author_email="y1qin@ucsd.edu",
    keywords="dexterous-manipulation point-cloud reinforcement-learning",
    description="DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yzqin/dexpoint_release",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "transforms3d",
        "gym==0.25.2",
        'sapien==2.1.0',
        "open3d>=0.15.2",
        "imageio",
        "torch>=1.11.0"
    ],
    extras_require={"tests": ["pytest", "black", "isort"]},
)
