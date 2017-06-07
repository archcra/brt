#!/usr/bin/env python
# -*- coding: utf-8 -*-

from struct import *
import numpy, sys

# http://stackoverflow.com/questions/25019287/how-to-convert-grayscale-values-in-matrix-to-an-image-in-python

# end if
imagesFile = 'data/brt-train-images02.data';
labelsFile = 'data/brt-train-labels02.data';

w, h = 70, 74 


# 前10张图表示的数字为：05 00 04 01 09 02 01 03 01 04
def getImagesData(): 
    with open(imagesFile, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        imagesCount = unpack(">i", fileContent[4:4+4])
        print (imagesCount[0])
        xMatrix = [[0 for x in range(70*74)] for y in range(imagesCount[0])] 

        # 5180 = 70*74
        for nthImage in range(1, imagesCount[0] + 1):
            imageBytes = unpack("5180c", fileContent[16+w*h*(nthImage-1):16+w*h*nthImage])
            
            for i in range(70*74):
                xMatrix[nthImage-1][i] = ord(imageBytes[i])
            # end for
        # end for
        
    return xMatrix

    
'''    
def getLabelsData(): 
    chessmenType = {
    'a': [0, 0, 0, 1],
    'b': [0, 0, 1, 0],
    'c': [0, 0, 1, 1],
    'k': [0, 1, 0, 0],
    'n': [0, 1, 0, 1],
    'p': [0, 1, 1, 0],
    'r': [0, 1, 1, 1],
    'B': [1, 0, 0, 0],
    'K': [1, 0, 0, 1],
    'A': [1, 0, 1, 0],
    'P': [1, 0, 1, 1]
    }

def getLabelsData():    
    chessmenType = {     
        'a': [0, 0, 1],
        'b': [0, 1, 0],
        'c': [0, 1, 1],
        'k': [1, 0, 0],
        'n': [1, 0, 1],
        'p': [1, 1, 0],
        'r': [1, 1, 1]   
        } 
'''
chessmenType = {
    'a': [1, 0, 0, 0, 0, 0, 0],
    'b': [0, 1, 0, 0, 0, 0, 0],
    'c': [0, 0, 1, 0, 0, 0, 0],
    'k': [0, 0, 0, 1, 0, 0, 0],
    'n': [0, 0, 0, 0, 1, 0, 0],
    'p': [0, 0, 0, 0, 0, 1, 0],
    'r': [0, 0, 0, 0, 0, 0, 1]
    }
    
def getLabelData(label):
    return chessmenType[label]
    
def getLabelsData(): 

    with open(labelsFile, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
        
        labelsCount = unpack(">i", fileContent[4:4+4])
        xMatrix = [0 for x in range(labelsCount[0])]  
        labelsBytes = unpack(str(labelsCount[0]) + "c", fileContent[8:8+labelsCount[0]])
        print(labelsBytes)
        for i in range(labelsCount[0]):
            # print('|'+labelsBytes[i]+'|')
            xMatrix[i] = chessmenType[str(chr(ord(labelsBytes[i])))]
        # end for
        
    return xMatrix
    
    
if __name__ == '__main__':
    # data = getImagesData()
    # print (data)    
    labels = getLabelsData()
    print (labels)
    

