from struct import *
from matplotlib import pyplot as plt
import numpy, sys
import loadApplyData

w, h = 70, 74 
data = loadApplyData.getImageData('data/pb5.png')
Matrix = [[0 for x in range(w)] for y in range(h)] 

for row in range(0, h):
    for col in range(0, w):
        # print row, col, row*w + col
        Matrix[row][col] = data[row*w + col]
        
# print Matrix
plt.imshow(Matrix, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
plt.savefig('./tmp.png')
plt.show()
# 通过！！！