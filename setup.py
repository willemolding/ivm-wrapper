import os
from setuptools import setup
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Customized setuptools install command - calls the matlab make file. This assumes that matlab is on your path"""
    def run(self):
        make_cmd = '''matlab -nodisplay -nosplash -nodesktop -r "run('./ivm/ivmSoftware4.3/src/make.m');exit;"'''
        os.system(make_cmd)
        install.run(self)

setup(
    name = "ivm-wrapper",
    version = "1.0.0",
    author = "Willem Olding",
    author_email = "willemolding@gmail.com",
    description = ("Python wrappers for the Matlab code for the Import Vector Machine (IVM) classifier from Freie University"),
    packages=['ivm'],
    install_requires=['scikit-learn', 'matlabengineforpython'],
    include_package_data=True,
    cmdclass={
        'install': CustomInstallCommand,
    }
)