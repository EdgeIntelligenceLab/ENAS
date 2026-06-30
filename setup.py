"""
Package configuration for ENAS-TinyML.
Authorship intentionally omitted for double-blind review.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="enas-tinyml",
    version="2.1.0",
    description=(
        "ENAS: Efficient Hardware-Aware Neural Architecture Search "
        "for TinyML Deployment"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anonymous",
    author_email="anonymous@example.com",
    url="<anonymous-for-review>",
    license="MIT",
    packages=find_packages(
        exclude=["tests", "tests.*", "notebooks", "datasets", "figures"]
    ),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
