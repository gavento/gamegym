import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gamegym",
    version="0.2.0",
    author="Tomáš Gavenčiak",
    author_email="gavento@gmail.com",
    description="Game theory framework, algorithms and game implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gavento/gamegym",
    packages=setuptools.find_packages(include=['gamegym', 'gamegym.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install with pip install -e .
    install_requires=[
        "numpy>=1.15",
        "attrs>=18.0",
        "tqdm>=4.28",
    ],
    # install with pip install -e .[dev]
    extras_require={
        'dev': [
            'codecov',
            'coverage>=4.0',
            'coveralls',
            'flake8>=3.0',
            'pylint>=2.0',
            'pytest-benchmark>=3.0',
            'pytest>=4.0',
            'tox-travis',
            'tox',
            'yapf',
        ]
    })
