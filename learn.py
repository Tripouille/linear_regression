import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils import *

if len(sys.argv) != 2 or not sys.argv[1].endswith('.csv'):
	sys.exit("usage: python3 learn.py *.csv")

with open(sys.argv[1]) as file:
	data = [line for i, line in enumerate(csv.reader(file)) if i]
x = np.asarray([float(line[0]) / 1E5 for line in data])
y = np.asarray([float(line[1]) / 1E5 for line in data])
x = x.reshape(x.shape[0], 1)
y = y.reshape(y.shape[0], 1)

try:
	with open("theta.npy") as file:
		theta = np.fromfile(file)
		theta = theta.reshape(theta.shape[0], 1)
except FileNotFoundError:	
	theta = np.zeros((2, 1))	

X = np.hstack((x, np.ones(x.shape)))
theta = gradient_descent(X, y, theta, 0.1, 1000);
plt.scatter(x, y, c='g', marker='+')
plt.plot(x, model(X, theta), c='b')

with open("theta.npy", 'w') as file:
	theta.tofile(file)
