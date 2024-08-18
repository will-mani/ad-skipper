import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import random

ad_images = ["ad3window.png", "ad4FULL.png", "ad5BRIGHT.png", "ad5DIM.png", "ad5mini.png", "ad6busy.png", "ad8DARK.png"]

def distance(point1, point2):
    horizontal_distance = abs(point1[0] - point2[0])
    vertical_distance = abs(point1[1] - point2[1])
    hypotenuse = (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5
    return hypotenuse

def equilateral_triangle(contour):
    #resizing the contour list so that a vertex/point is simply
    # a list of length 2 rather than a nested list
    contour_array = np.resize(contour, (len(contour), 2))
    x_values_list = contour_array[:, 0].tolist()
    y_values_list = contour_array[:, 1].tolist()

    top_left_index = y_values_list.index(max(y_values_list))
    triangle_vertex1 = contour_array[top_left_index]
    bottom_left_index = y_values_list.index(min(y_values_list))
    triangle_vertex2 = contour_array[bottom_left_index]
    far_right_index = x_values_list.index(max(x_values_list))
    triangle_vertex3 = contour_array[far_right_index]

    side1_length = distance(triangle_vertex1, triangle_vertex2)
    side2_length = distance(triangle_vertex1, triangle_vertex3)
    side3_length = distance(triangle_vertex2, triangle_vertex3)
    if abs(side1_length - side2_length) / max(side1_length, side2_length) > 0.3:
        return False
    elif abs(side1_length - side3_length) / max(side1_length, side3_length) > 0.3:
        return False
    elif abs(side2_length - side3_length) / max(side2_length, side3_length) > 0.3:
        return False
    
    return True
    

for ad_image in ad_images:
    image_path = "C://Users//willi//Desktop//Ad skrnshts//" + ad_image
	
    image = cv2.imread(image_path) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY) 

    triangles = np.zeros(image.shape, dtype=np.uint8)

    # using findContours() function 
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    
    for i in range(len(contours)): 
        contour = contours[i]
        # we ignore the first contour because the
        # cv2.findContours() detects the whole image as a shape 
        if i == 0:
            continue

        # cv2.approxPloyDP() approximates the shape 
        approx = cv2.approxPolyDP( 
            contour, 0.05 * cv2.arcLength(contour, True), True) 
        
        # checking whether the approximted tirangle is equilateral 
        if len(approx) == 3:
            if equilateral_triangle(contour):
                # print(contour)
                # print("-----------")
                cv2.drawContours(image, [contour], 0, (0, 0, 255), 1) 
                cv2.drawContours(triangles, [contour], 0, (0, 255, 0), -1) 


    # displaying the images 
    factor = 0.7
    cv2.imshow('gray', cv2.resize(gray, (0,0), fx= factor, fy = factor)) 
    cv2.imshow('threshold', cv2.resize(threshold, (0,0), fx= factor, fy = factor)) 
    cv2.imshow('triangles', cv2.resize(triangles, (0,0), fx= factor, fy = factor))
    cv2.imshow('image', cv2.resize(image, (0,0), fx= factor, fy = factor)) 

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
