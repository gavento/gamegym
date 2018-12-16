import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gamegym",
    version="0.1.1",
    author="Tomáš Gavenčiak",
    author_email="gavento@gmail.com",
    description="Game theory framework, algorithms and game implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gavento/gamegym",
    packages=["gamegym"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install with pip install -e .
    install_requires = [
        "numpy>=1.15",
        "attrs>=18.0",
        "tqdm>=4.28",
    ],
    # install with pip install -e .[dev]
    extras_require={
        'dev': [
            'coverage',
            'pylint',
            'pytest-benchmark',
            'pytest',
            'flake8',
            'yapf',
        ]
    }

)
