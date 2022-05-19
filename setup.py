from setuptools import find_packages, setup

setup(
    name="eagerx_quadruped",
    packages=[package for package in find_packages() if package.startswith("eagerx_quadruped")],
    # package_data={"sb3_contrib": ["py.typed", "version.txt"]},
    install_requires=["eagerx>=0.1.24", "eagerx-pybullet>=0.1.8"],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
    },
    description="",
    author="Antonin Raffin",
    url="",
    author_email="antonin.raffin@dlr.de",
    license="MIT",
    long_description="",
    long_description_content_type="text/markdown",
    version="0.1.2",
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
