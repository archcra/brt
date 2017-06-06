#!/usr/bin/env python
# -*- coding: utf-8 -*-




from struct import *
from PIL import Image

h, w = 74, 70

def filterAndRevert(img):
    for row in range(0, h): # height is 1280
        for col in range(0,w):
            pixel = img.getpixel((col, row))
            if(pixel > 140): # 这个值与测试图片组有关
                img.putpixel((col, row), 0)
            else:
                img.putpixel((col, row),255 -pixel )
            # end if
        # end inner for
    # end outer for
    
    # resize
    smallImage = img.resize((wResized,hResized))
    smallImage.save('output/small.png')
    return smallImage
    
    

imageFileName = "output/brt-train-images.data"
labelFileName = "output/brt-train-labels.data"

imagesPath = "/Users/papa/vcs/github/brt/scaffold/buildTtDataset/chessmen/"

imageDataFile=open(imageFileName,'wb')
labelDataFile=open(labelFileName,'wb')


# 写magic number: 4 bytes
imageDataFile.write(pack('>i', 20170606))
labelDataFile.write(pack('>i', 20170606))

# Read all images from a foler
import os
imageFiles = os.listdir(imagesPath)
print (imageFiles)

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
    imageNameWithoutExt= imageFiles[i][:len(imageFiles[i])-4]
    label = imageNameWithoutExt[0:1]
    print ("Label is: " +  label)
    
    # write the label
    labelDataFile.write(pack('B', ord(label)-48))
    
    img = Image.open(imagesPath+imageFiles[i]).convert('L') 
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









