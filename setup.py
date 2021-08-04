import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="framegraph",
    version="0.0.1",
    description="A generic graph structure representing relative transforms between coordinate systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vi-robotics/framegraph",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'numpy-quaternion',
    ],
    extras_require={
        'debug': [
            'pyvista'
        ]
    },
    python_requires='>=3.6',
)
