from setuptools import setup, find_packages

setup(
    name="hmp_utils",
    version="0.0.1",
    author="Michael Vanuzzo",
    author_email="michaelvanuzzo@gmail.com",
    packages=find_packages(),
    # package_dir={'': 'src'},
    # url='https://github.com/.../...',
    # keywords='example project',
    install_requires=[
        # //examples:
        # 'numpy',
        # 'torch',
    ],
    package_data={
        "hmp_utils": ["pds_files/*.npy", "amass_split.json"],
    },
)
