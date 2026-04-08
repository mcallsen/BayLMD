import setuptools

setuptools.setup(
    name="forcefieldml",
    version="1.0.0",
    author="Martin Callsen",
    author_email="mcallsen@as.edu.tw",
    description="A small example package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=("./tests",)),
    python_requires=">=3.9",
)
