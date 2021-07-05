import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils import *
ITERATONS = 10000
LEARNING_RATE = 0.1

if len(sys.argv) != 2 or not sys.argv[1].endswith('.csv'):
	sys.exit("usage: python3 learn.py *.csv")

try:
	with open(sys.argv[1]) as file:
		data = [line for line in csv.reader(file)]
		x = np.asarray([float(line[0]) / 1E5 for line in data[1:]])
		y = np.asarray([float(line[1]) / 1E5 for line in data[1:]])
except FileNotFoundError: sys.exit("Error: can't found file " + sys.argv[1])
except ValueError: sys.exit("Error: corrupted file.")

x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

theta = get_theta()
X = np.hstack((x, np.ones(x.shape)))
theta, cost_history = gradient_descent(X, y, theta, LEARNING_RATE, ITERATONS)
plt_infos = plt.figure(1)
plt.scatter(x, y, c='g', marker='+')
plt.plot(x, model(X, theta), c='b')

with open(".theta.npy", 'w') as file:
	theta.tofile(file)

plt.suptitle("linear_regression")
plt.title("coef = " + str(get_coef_determination(y, model(X, theta))))
plt.xlabel(data[0][0] + " / 1E5")
plt.ylabel(data[0][1] + " / 1E5")

plt_cost = plt.figure(2)
plt.suptitle("cost evolution")
plt.title("learning rate = " + str(LEARNING_RATE))
plt.xlabel("iterations")
plt.ylabel("cost")
plt.plot(range(ITERATONS), cost_history)
plt.show()