from setuptools import setup, find_packages

scripts = [
    "oitg_fetch=oitg.frontend.oitg_fetch:main",
    "oitg_index_rids=oitg.frontend.oitg_index_rids:main",
]

setup(
    name='oitg',
    version='0.1',
    packages=find_packages(),
    install_requires=['statsmodels'],
    entry_points={"console_scripts": scripts},
)
