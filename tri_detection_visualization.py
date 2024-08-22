import triangle_detector
import cv2
import numpy as np 

ad_images = ["ad3window.png", "ad4FULL.png", "ad5BRIGHT.png", "ad5DIM.png", "ad5mini.png", "ad6busy.png", "ad8DARK.png"]

for ad_image in ad_images:
    image_path = "C://Users//willi//Desktop//Ad skrnshts//" + ad_image
	
    image = cv2.imread(image_path) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY) 
    triangles = np.zeros(image.shape, dtype=np.uint8)

    triangle_vertices_list = triangle_detector.find_triangle_vertices(image_path)

    for triangle_vertices in triangle_vertices_list: 
        vertices_array = np.array(triangle_vertices)
        cv2.polylines(image, [vertices_array], True, (0, 0, 255), 5) 
        cv2.fillPoly(triangles, [vertices_array], (0, 255, 0)) 


    # displaying the images 
    factor = 0.7
    cv2.imshow('gray', cv2.resize(gray, (0,0), fx= factor, fy = factor)) 
    cv2.imshow('threshold', cv2.resize(threshold, (0,0), fx= factor, fy = factor)) 
    cv2.imshow('triangles', cv2.resize(triangles, (0,0), fx= factor, fy = factor))
    cv2.imshow('image', cv2.resize(image, (0,0), fx= factor, fy = factor)) 

    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
