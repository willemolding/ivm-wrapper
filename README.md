ivm-wrapper
===========

Python wrappers for the Matlab code for the Import Vector Machine (IVM) classifier from Freie University (http://www.ipb.uni-bonn.de/ivm/?L=1)

Install
-------

This assumes the following:
- you already have installed the MATLAB engine for python (http://au.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- The Matlab executable is on your path
- You have configured mex to work correctly

Installation Steps:
- Clone the github or download the source code
- Download the ivmSoftware4.3 from (http://www.geo.fu-berlin.de/en/geog/fachrichtungen/geoinformatik/medien/download/ivmSoftware4_3.zip)
- Extract the zip file to ivm-wrapper/ivm/ivmSoftware4.3
- run 'python setup.py install'

Example
-------

The wrapper extends scikit-learn base classifier so it should be compatible with all the scikit-learn extras such as grid_search and cross_val_score.
A simple test on the iris dataset:

	from sklearn import datasets
	from sklearn.cross_validation import cross_val_score
	from ivm import IVM

	data = datasets.load_iris()
	clf = IVM()
	print cross_val_score(clf, data.data, data.target)