import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import random

images = ["ad3window.png", "ad4FULL.png", "ad5BRIGHT.png", "ad5DIM.png", "ad5mini.png", "ad6busy.png", "ad8DARK.png"]

for image in images:
    image_path = "Ad skrnshts//" + image
	
    img = cv2.imread(image_path) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) 

    blank = np.zeros(gray.shape, dtype=np.uint8)

    # using a findContours() function 
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    i = 0

    # list for storing names of shapes 
    for contour in contours: 

        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape 
        if i == 0: 
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape 
        approx = cv2.approxPolyDP( 
            contour, 0.05 * cv2.arcLength(contour, True), True) 
        
        # using drawContours() function 
        if len(approx) == 3:
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 1) 
            cv2.drawContours(blank, [contour], 0, (255, 255, 255), -1) 


    # displaying the image after drawing contours 
    factor = 0.7
    cv2.imshow('gray', cv2.resize(gray, (0,0), fx= factor, fy = factor)) 
    cv2.imshow('thresh', cv2.resize(threshold, (0,0), fx= factor, fy = factor)) 
    cv2.imshow('blank canvas', cv2.resize(blank, (0,0), fx= factor, fy = factor))
    cv2.imshow('shapes', cv2.resize(img, (0,0), fx= factor, fy = factor)) 

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
