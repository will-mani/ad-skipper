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
    x_values_array = contour_array[:, 0]
    y_values_array = contour_array[:, 1]

    top_left_index = y_values_array.argmin()
    triangle_vertex1 = contour_array[top_left_index].tolist()
    bottom_left_index = y_values_array.argmax()
    triangle_vertex2 = contour_array[bottom_left_index].tolist()
    far_right_index = x_values_array.argmax()
    triangle_vertex3 = contour_array[far_right_index].tolist()

    triangle_vertices = [triangle_vertex1, triangle_vertex2, triangle_vertex3]

    side1_length = distance(triangle_vertex1, triangle_vertex2)
    side2_length = distance(triangle_vertex1, triangle_vertex3)
    side3_length = distance(triangle_vertex2, triangle_vertex3)
    
    if side1_length == 0 or side2_length == 0 or side3_length == 0:
        return False
    # checking whether the sides are roughly equilateral
    # (difference of at most 30% of the longer side)
    elif abs(side1_length - side2_length) / max(side1_length, side2_length) > 0.3:
        return False
    elif abs(side1_length - side3_length) / max(side1_length, side3_length) > 0.3:
        return False
    elif abs(side2_length - side3_length) / max(side2_length, side3_length) > 0.3:
        return False
    # checking triangle orientation (should like an arrow pointing right)
    elif abs(triangle_vertex1[0] - triangle_vertex2[0]) / side1_length > 0.05:
        return False
    # making sure the potential triangle is not too small
    elif cv2.contourArea(np.array(triangle_vertices)) < 10:
        return False
    
    triangle_vertices_list.append(triangle_vertices)
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

        # cv2.approxPloyDP() approximates the number of sizes of the contour
        perimerter = cv2.arcLength(contour, True)
        approx_sides = cv2.approxPolyDP( 
            contour, 0.05 * perimerter, True) 
        
        # checking whether the approximted tirangle is equilateral 
        if len(approx_sides) == 3:
            right_arrow_triangle(contour, triangle_vertices_list)

    return triangle_vertices_list