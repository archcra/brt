#!/usr/bin/env python
# -*- coding: utf-8 -*-

from struct import *
import numpy, sys
from PIL import Image


h, w = 74, 70

def getImageData(imageFilePath): 
    # imageFilePath = 'data/pb5.png'
    img = Image.open(imageFilePath).convert('L') 
    imgFiltered = img # filterAndRevert(img) 不做过滤先
    xMatrix = [0 for x in range(70*74)]
    i = 0
    for row in range(0, h):
        for col in range(0,w):
            print ('row is:' + str(row) + ' col is: ' + str(col))
            pixel = imgFiltered.getpixel((col, row)) # x is col, y is row
            xMatrix[i] = pixel
            i = i + 1

    return xMatrix    

if __name__ == '__main__':
    data = getImageData('data/pb5.png')
    print (data)

