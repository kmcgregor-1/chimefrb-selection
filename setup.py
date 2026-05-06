from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="chimefrb-selection",
    version="0.1.0",
    description="Selection function modeling & masks for CHIME/FRB injections",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Kyle McGregor",
    license="MIT",
    url="https://github.com/CHIMEFRB/chimefrb-selection",
    project_urls={
        "Homepage": "https://github.com/CHIMEFRB/chimefrb-selection",
        "Issues": "https://github.com/CHIMEFRB/chimefrb-selection/issues",
    },

    # Package discovery: current layout has the package at ./chimefrb_selection
    packages=find_packages(exclude=("tests", "docs", "examples")),

    # Python & runtime deps (aligned with your TOML)
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "scikit-learn>=1.0",
        "matplotlib>=3.5",
        "click>=8",
        "h5py>=3.7",
    ],

    # Include model npz/pkl files inside the wheel
    include_package_data=True,
    package_data={
        "chimefrb_selection": [
            "data/*",
            "data/**/*",
            "data/fits/*.npz",
            "data/fits/**/*.npz",
            "data/masks/*.pkl",
            "data/*README*.md",
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    zip_safe=False,
)
