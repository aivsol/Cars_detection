import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import math

# imR = cv2.imread(im_fileR)
# imL = cv2.imread(im_fileL)


B = 0.164
X = 2048
theta = 69.4
disparity = 910.45-742.96

D = (B*X)/(math.tan(theta/2*math.pi/180)*disparity)
print D