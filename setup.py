from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="timeseriesutils",
    version="0.1.0", 
    author="Davide Vidotto",
    description="A package for time series utilities and forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidevdt/timeseriesutils",  
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6', 
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "pykalman>=0.9.5",
    ],
    entry_points={
        "console_scripts": [
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/davidevdt/timeseriesutils/issues",
        "Source": "https://github.com/davidevdt/timeseriesutils",
    },
)
