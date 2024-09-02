import cv2 
import numpy as np 

class triangle_detector():
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) 
        #_, self.threshold_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY) 
        self.threshold_image = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5)
        self.triangles_list = [] # list of all triangles, where each entry is a list of the triangle vertices

    def find_triangles(self):
        # Find contours of the (purely) black and white threshold image
        contours, _ = cv2.findContours(self.threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[1:]: # we ignore the first contour since cv2.findContours() detects the whole image as a shape 
            # cv2.approxPloyDP() approximates the number of sides of the contour
            perimerter = cv2.arcLength(contour, True)
            approx_sides = cv2.approxPolyDP(contour, 0.05 * perimerter, True) 
            # checking whether the approximted tirangle looks like an arrow pointing right
            if len(approx_sides) == 3:
                self.right_arrow_triangle(contour)

        return self.triangles_list

    def right_arrow_triangle(self, contour):
        # resizing the contour list so that a vertex/point is simply a list of length 2 rather than a nested list
        contour_array = np.resize(contour, (len(contour), 2))
        # contour points are (x, y) pairs, not (y, x) / (row, col) like numpy
        x_values_array = contour_array[:, 0]
        y_values_array = contour_array[:, 1]

        top_left_index = y_values_array.argmin()
        top_left_vertex = contour_array[top_left_index].tolist()
        bottom_left_index = y_values_array.argmax()
        bottom_left_vertex = contour_array[bottom_left_index].tolist()
        far_right_index = x_values_array.argmax()
        far_right_vertex = contour_array[far_right_index].tolist()

        triangle_vertices = [top_left_vertex, bottom_left_vertex, far_right_vertex]

        side1_length = distance(top_left_vertex, bottom_left_vertex)
        side2_length = distance(top_left_vertex, far_right_vertex)
        side3_length = distance(bottom_left_vertex, far_right_vertex)
        
        if side1_length == 0 or side2_length == 0 or side3_length == 0:
            return False
        # checking whether the sides are roughly equilateral (difference of at most 30% of the longer side)
        elif max_side_length_ratio(side1_length, side2_length, side3_length) > 0.3:
            return False
        # checking triangle orientation (should like an arrow pointing right)
        elif abs(top_left_vertex[0] - bottom_left_vertex[0]) / side1_length > 0.1:
            return False
        # making sure the potential triangle is not too small
        elif cv2.contourArea(np.array(triangle_vertices)) < 10:
            return False
        # making sure that the potential triangle is mostly white in the threshold image
        elif self.white_fill_percentage(triangle_vertices) < 70:
            return False
        # checking if the contour looks about the same as the potential "right arrow" triangle
        elif self.overlap_percentage(contour, triangle_vertices) < 70:
            return False
        
        self.triangles_list.append(triangle_vertices)
        return True
    
    def white_fill_percentage(self, triangle_vertices):
        triangle_image = np.zeros(self.threshold_image.shape, dtype=np.uint8)
        vertices_array = np.array(triangle_vertices)
        cv2.fillPoly(triangle_image, [vertices_array], 255)
        
        triangle_coors = np.where(triangle_image == 255) 
        triangle_crop = self.threshold_image[triangle_coors] # triangular crop of threshold image
        white_pixel_coors = np.where(triangle_crop == 255)
        fill_percentage = white_pixel_coors[0].size / triangle_crop.size * 100
        return fill_percentage

    
    def overlap_percentage(self, contour, triangle_vertices):
        contour_image = np.zeros(self.threshold_image.shape, dtype=np.uint8)
        triangle_image = np.ones(self.threshold_image.shape, dtype=np.uint8) # if this were also np.zeros, it would cause trouble with finding overlaps later on
        cv2.drawContours(contour_image, [contour], 0, 255, -1) # 0 = index, 255 = white, -1 = fill
        vertices_array = np.array(triangle_vertices)
        cv2.drawContours(triangle_image, [vertices_array], 0, 255, -1) # fill has to be the same as the contour image for the next part to work
        
        overlapping_coors = np.where(contour_image - triangle_image == 0) # works because the difference is 0 only when the drawn filled drawings overlap 
        contour_coors = np.where(contour_image == 255)
        triangle_coors = np.where(triangle_image == 255) 
        percentage = overlapping_coors[0].size / max(contour_coors[0].size, triangle_coors[0].size) * 100
        return percentage

    
def distance(point1, point2):
    horizontal_distance = abs(point1[0] - point2[0])
    vertical_distance = abs(point1[1] - point2[1])
    hypotenuse = (horizontal_distance ** 2 + vertical_distance ** 2) ** 0.5
    return hypotenuse

def max_side_length_ratio(side_length1, side_length2, side_length3):
    ratio1 = abs(side_length1 - side_length2) / max(side_length1, side_length2)
    ratio2 = abs(side_length1 - side_length3) / max(side_length1, side_length3)
    ratio3 = abs(side_length2 - side_length3) / max(side_length2, side_length3)
    return max(ratio1, ratio2, ratio3)