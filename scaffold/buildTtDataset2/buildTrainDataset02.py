#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This can use any images with name pattern: rb001.png
# first letter is used in FIN, r is rook, b is color: black


from struct import *
from PIL import Image
import glob

h, w = 74, 70
    
imageFileName = "output/brt-train-images02.data"
labelFileName = "output/brt-train-labels02.data"

imagesPath = "/Users/papa/vcs/github/brt/scaffold/buildTtDataset2/chessmen/"
imagesFilter = imagesPath + "*.png"

imageDataFile=open(imageFileName,'wb')
labelDataFile=open(labelFileName,'wb')


# 写magic number: 4 bytes
imageDataFile.write(pack('>i', 20170606))
labelDataFile.write(pack('>i', 20170606))

# Read all images from a foler
# import os
# imageFiles = os.listdir(imagesPath)
imageFiles = glob.glob(imagesFilter)

# write the count# 写count: 4 bytes
imageDataFile.write(pack('>i', len(imageFiles)))
labelDataFile.write(pack('>i', len(imageFiles)))


# Write the image width and height
# 写width, height
imageDataFile.write(pack('>i', w))

# 写columns
imageDataFile.write(pack('>i', h))


# Process each file
for i in range(0, len(imageFiles)):
    # First, convert image to grayscale 
    # '/Users/papa/vcs/github/brt/scaffold/buildTtDataset2/chessmen/ab001.png'
    imageNameWithoutExt= imageFiles[i][len(imageFiles[i])-9:]
    label = imageNameWithoutExt[0:1]
    print ("Label is: " +  label)
    
    # write the label
    labelDataFile.write(pack('B', ord(label[:1])))
    
    img = Image.open(imageFiles[i]).convert('L') 
    # not LA , which means alpha; 这里转成灰度图
    
    imgFiltered = img # filterAndRevert(img) 不做过滤先
    for row in range(0, h):
        for col in range(0,w):
            print ('row is:' + str(row) + ' col is: ' + str(col))
            pixel = imgFiltered.getpixel((col, row)) # x is col, y is row
            imageDataFile.write(pack('B', pixel))

# close the file handler
imageDataFile.close()
labelDataFile.close()









