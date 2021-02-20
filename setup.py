from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="emmv",
    version="0.0.0",
    author="Christian O'Leary",
    author_email="christian.oleary@cit.ie",
    description='Metrics for unsupervised anomaly detection models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://gitlab.com/chris.oleary/emmv',
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
