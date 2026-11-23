from setuptools import setup, find_packages

setup(
    name="hmp_utils",
    version="0.1.0",
    author="Michael Vanuzzo",
    author_email="michaelvanuzzo@gmail.com",
    description="Human Motion Prediction Utilities",
    packages=find_packages(),
    license="GNU Affero General Public License v3 (AGPL-3.0)",
    license_files=("LICENSE",),
    url="https://github.com/MichaelVanuzzo/hmp_utils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "smplx",
        "scikit-learn",
        "tqdm",
        "pillow",
        "imageio",
        "ipywidgets",
    ],
    package_data={
        "hmp_utils": ["pds_files/*.npy", "amass_split.json"],
    },
    python_requires=">=3.6",
)
