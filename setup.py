import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gamegym",
    version="0.1.0",
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
)
