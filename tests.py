from ivm import IVM
from sklearn import datasets
from unittest import TestCase
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score


class TestIVM(TestCase):

	def setUp(self):
		self.clf = IVM()

	def test_get_params(self):
		self.clf.get_params()

	def test_set_params(self):
		self.clf.set_params(_lambda=3)

	def test_fit_precict(self):
		data = datasets.load_iris()
		print cross_val_score(self.clf, data.data, data.target)