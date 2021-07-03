import sys
import numpy as np
from utils import *

if len(sys.argv) != 2:
	sys.exit("usage: python3 predict.py km")

try:
	with open("theta.npy") as file:
		theta = np.fromfile(file)
		theta = theta.reshape(theta.shape[0], 1)
except FileNotFoundError:	
	theta = np.zeros((2, 1))

value = np.asarray([float(sys.argv[1]) / 1E5, 1.0])
print(model(value, theta)[0] * 1E5)