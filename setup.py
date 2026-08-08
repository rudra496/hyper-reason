from setuptools import setup, find_packages

setup(
    name="hyper-reason",
    version="1.0.0",
    description="Autonomous Test-Time Compute Scaling & Dynamic KV-Cache Sparsification for Edge & Local LLMs",
    long_description=open("README.txt", encoding="utf-8").read(),
    long_description_content_type="text/plain",
    author="Rudra Sarker",
    author_email="rudrasarker130@gmail.com",
    url="https://github.com/rudra496/hyper-reason",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hyper-reason=hyper_reason.cli:main",
        ],
    },
)
