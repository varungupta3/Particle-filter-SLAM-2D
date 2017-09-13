import numpy as np
import copy

class SE2:
	""" A class for special euclidian group of dimensionality 2
	"""
	def __init__(self, p, theta):
		self.p = p
		self.theta = theta

	@classmethod
	def from_foo(cls, class_instance):
		p = copy.deepcopy(class_instance._p) # if deepcopy is necessary
		theta = copy.deepcopy(class_instance._theta)
		return cls(p, theta)


	def __add__(self, other):
		R = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
		p = self.p + np.matmul(R, other.p)
		theta = self.theta + other.theta
		return SE2(p, theta)

	def __sub__(self, other): # self.thetas in R ?!
		R = np.array([[np.cos(other.theta), -np.sin(other.theta)], [np.sin(other.theta), np.cos(other.theta)]])
		p = np.matmul(R.T, self.p - other.p)
		theta = self.theta - other.theta
		return SE2(p, theta)

	def inverse(self):
		theta = -self.theta
		R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
		p = np.matmul(-R, self.p)
		return SE2(p, theta)



