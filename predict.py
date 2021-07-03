import sys
import numpy as np
from utils import *

if len(sys.argv) != 2:
	sys.exit("usage: python3 predict.py km")

theta = get_theta()
value = np.asarray([float(sys.argv[1]) / 1E5, 1.0])
print(model(value, theta)[0] * 1E5)