ivm-wrapper
===========

Python wrappers for the Matlab code for the Import Vector Machine (IVM) classifier from Freie University (http://www.ipb.uni-bonn.de/ivm/?L=1)

Install
=======

This assumes the following:
- you already have installed the MATLAB engine for python (http://au.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- The Matlab executable is on your path
- You have configured mex to work correctly

Installation Steps:
- Clone the github or download the source code
- Download the ivmSoftware4.3 from (http://www.geo.fu-berlin.de/en/geog/fachrichtungen/geoinformatik/medien/download/ivmSoftware4_3.zip)
- Extract the zip file to ivm-wrapper/ivm/ivmSoftware4.3
- run 'python setup.py install'