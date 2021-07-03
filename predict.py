import sys
import numpy as np
from utils import *

value = np.asarray([float(input("Please enter km: ")) / 1E5, 1.0])
print(model(value, get_theta())[0] * 1E5)