import numpy as np

def model(X, theta):
	return (X.dot(theta))

def cost(X, y):
	return (1 / (2 * len(y)) * np.sum((X - y) ** 2))

def gradient(X, y, theta):
	return (1 / len(y) * X.T.dot(model(X, theta) - y))

def gradient_descent(X, y, theta, learning_rate, it):
	for _ in range(it):
		theta -= learning_rate * gradient(X, y, theta)
	return theta

def get_theta():
	try:
		with open(".theta.npy") as file:
			theta = np.fromfile(file)
			theta = theta.reshape(theta.shape[0], 1)
	except FileNotFoundError:	
		theta = np.zeros((2, 1))
	return (theta)