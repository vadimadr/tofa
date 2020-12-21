from pathlib import Path

from setuptools import setup, find_packages

NAME = "tofa"
DESCRIPTION = "Collection of requisites to speed up research & development with PyTorch"
AUTHOR = "Vadim Andronov"
EMAIL = "vadimadr@gmail.com"
URL = "https://github.com/vadimadr/tofa"
LICENSE = "BSD-3"
KEYWORDS = [
    "PyTorch",
    "Computer Vision",
    "Deep Learning",
    "Machine Learning",
]

SOURCE_ROOT = Path(__file__).parent.resolve()


def read_file(file_name):
    with file_name.open("r") as f:
        return f.read()


requirements = read_file(SOURCE_ROOT / "requirements.txt")

# add opencv if needed. Avoids reinstalling opencv if already build from source
try:
    import cv2
except ImportError:
    requirements.append("opencv-python")

metadata = {}
with Path(__file__).parent.joinpath("tofa", "__version__.py").open("r") as f:
    exec(f.read(), metadata)


setup(
    name=NAME,
    description=DESCRIPTION,
    long_description=read_file(SOURCE_ROOT / "README.md"),
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(),
    requirements=requirements,
    python_requires=">=3.6",
    keywords=KEYWORDS,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    version=metadata["__version__"],
    project_urls={"Source": "https://github.com/vadimadr/tofa"},
)
