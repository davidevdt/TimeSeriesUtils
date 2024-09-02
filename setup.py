from setuptools import setup, find_packages

setup(
    name='TimeSeriesUtils',
    version='0.1.0',
    description='A utility package for time series analysis and forecasting.',
    long_description=open('README.md').read(),  # Ensure you have a README.md file for the long description
    long_description_content_type='text/markdown',
    author='Davide Vidotto',
    url='https://github.com/davidevdt/TimeSeriesUtils',  # Replace with your repository URL
    packages=find_packages(),  # Automatically find packages in the current directory
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'scikit-learn>=0.24.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'sphinx>=4.0.0',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Specify the minimum Python version required
    include_package_data=True,
    zip_safe=False,
)
