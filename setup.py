from setuptools import setup
import pkg_resources

setup(
    name = "ivm-wrapper",
    version = "1.0.0",
    author = "Willem Olding",
    author_email = "willemolding@gmail.com",
    description = ("Python wrappers for the Matlab code for the Import Vector Machine (IVM) classifier from the University of Bonn"),
    packages=['ivm'],
    install_requires=['scikit-learn', 'matlabengineforpython'],
    include_package_data=True
)