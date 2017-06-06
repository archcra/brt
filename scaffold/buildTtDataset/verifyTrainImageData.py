#!/usr/bin/env python
# -*- coding: utf-8 -*-

from struct import *
from matplotlib import pyplot as plt
import numpy, sys

# http://stackoverflow.com/questions/25019287/how-to-convert-grayscale-values-in-matrix-to-an-image-in-python

print (len(sys.argv))
if len(sys.argv) != 2:
    print('Usage: python3 verifyTrainImageData.py n , n是一个数字，1到7')
    exit(0)
# end if
imagesFile = 'output/brt-train-images.data';

w, h = 70, 74 
Matrix = [[0 for x in range(w)] for y in range(h)] 

nthImage = int(sys.argv[1]) # 5 # 3 # 这里的数字即第n个图，从1到10
# 前10张图表示的数字为：05 00 04 01 09 02 01 03 01 04

with open(imagesFile, mode='rb') as file: # b is important -> binary
    fileContent = file.read()
    # 5180 = 70*74
    imageBytes = unpack("5180c", fileContent[16+w*h*(nthImage-1):16+w*h*nthImage])
    
    for row in range(0, h):
        for col in range(0, w):
            # print row, col, row*w + col
            Matrix[row][col] = ord(imageBytes[row*w + col])
    # print Matrix
    plt.imshow(Matrix, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    plt.savefig('output/tmp.png')
    plt.show()
    
    
    # verify passed
    
    
    
    
    
    

    

