import cv2 
import numpy as np 

def distance(point1, point2):
    horizontal_distance = abs(point1[0] - point2[0])
    vertical_distance = abs(point1[1] - point2[1])
    hypotenuse = (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5
    return hypotenuse

def right_arrow_triangle(contour, triangle_vertices_list):
    # resizing the contour list so that a vertex/point is simply
    # a list of length 2 rather than a nested list
    contour_array = np.resize(contour, (len(contour), 2))
    x_values_list = contour_array[:, 0].tolist()
    y_values_list = contour_array[:, 1].tolist()

    top_left_index = y_values_list.index(max(y_values_list))
    triangle_vertex1 = contour_array[top_left_index].tolist()
    bottom_left_index = y_values_list.index(min(y_values_list))
    triangle_vertex2 = contour_array[bottom_left_index].tolist()
    far_right_index = x_values_list.index(max(x_values_list))
    triangle_vertex3 = contour_array[far_right_index].tolist()

    side1_length = distance(triangle_vertex1, triangle_vertex2)
    side2_length = distance(triangle_vertex1, triangle_vertex3)
    side3_length = distance(triangle_vertex2, triangle_vertex3)
    
    if side1_length == 0 or side2_length == 0 or side3_length == 0:
        return False
    # checking whether the sides are approximately equilateral
    elif abs(side1_length - side2_length) / max(side1_length, side2_length) > 0.3:
        return False
    elif abs(side1_length - side3_length) / max(side1_length, side3_length) > 0.3:
        return False
    elif abs(side2_length - side3_length) / max(side2_length, side3_length) > 0.3:
        return False
    # checking triangle orientation (should like an arrow pointing right)
    elif abs(triangle_vertex1[0] - triangle_vertex2[0]) / side1_length > 0.05:
        return False
    
    triangle_vertices_list.append([triangle_vertex1, triangle_vertex2, triangle_vertex3])
    return True
    
def find_triangle_vertices(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY) 

    # using findContours() function 
    contours, _ = cv2.findContours( 
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    triangle_vertices_list = []
    
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
            right_arrow_triangle(contour, triangle_vertices_list)

    return triangle_vertices_list