from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="emmv",
    version="0.0.4",
    author="Christian O'Leary",
    author_email="cjjoleary@gmail.com",
    description='Metrics for unsupervised anomaly detection models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/christian-oleary/emmv',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'setuptools',
        'scikit-learn',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
