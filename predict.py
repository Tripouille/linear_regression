import sys
import numpy as np
from utils import *

try: value = np.asarray([float(input("Please enter km: ")) / 1E5, 1.0])
except ValueError: sys.exit("Error: invalid input.")
print(model(value, get_theta())[0] * 1E5)