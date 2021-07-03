import numpy as np

def model(X, theta):
	return (X.dot(theta))

def cost(y, p):
	return (1 / (2 * len(y)) * np.sum((p - y) ** 2))

def gradient(X, y, theta):
	return (1 / len(y) * X.T.dot(model(X, theta) - y))

def gradient_descent(X, y, theta, learning_rate, it):
	cost_history = np.zeros(it)
	for i in range(it):
		cost_history[i] = cost(y, model(X, theta))
		theta -= learning_rate * gradient(X, y, theta)
	return (theta, cost_history)

def get_theta():
	try:
		with open(".theta.npy") as file:
			theta = np.fromfile(file)
			theta = theta.reshape(theta.shape[0], 1)
	except FileNotFoundError:	
		theta = np.zeros((2, 1))
	return (theta)

def get_coef_determination(y, p):
	return (1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())