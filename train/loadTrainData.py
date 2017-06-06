#!/usr/bin/env python
# -*- coding: utf-8 -*-

from struct import *
import numpy, sys

# http://stackoverflow.com/questions/25019287/how-to-convert-grayscale-values-in-matrix-to-an-image-in-python

# end if
imagesFile = 'data/brt-train-images.data';

w, h = 70, 74 
chessmenTypesCount = 7
xMatrix = [[0 for x in range(70*74)] for y in range(chessmenTypesCount)] 

# 前10张图表示的数字为：05 00 04 01 09 02 01 03 01 04
def getImagesData(): 
    with open(imagesFile, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        
        
        # 5180 = 70*74
        for nthImage in range(1, chessmenTypesCount + 1):
            imageBytes = unpack("5180c", fileContent[16+w*h*(nthImage-1):16+w*h*nthImage])
            
            for i in range(70*74):
                xMatrix[nthImage-1][i] = ord(imageBytes[i])
            # end for
        # end for
        
    return xMatrix

    
    
def getLabelsData(): 
    with open(imagesFile, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        # 5180 = 70*74
        for nthImage in range(1, chessmenTypesCount + 1):
            imageBytes = unpack("5180c", fileContent[16+w*h*(nthImage-1):16+w*h*nthImage])
            
            for i in range(70*74):
                xMatrix[nthImage-1][i] = ord(imageBytes[i])
            # end for
        # end for
        
    return xMatrix
    
    
    
if __name__ == '__main__':
    data = getImagesData()
    print (data)    

    

