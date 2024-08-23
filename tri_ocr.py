from PIL import Image
from pytesseract import pytesseract
import cv2 
import pandas as pd
import numpy as np
import triangle_detector

path_to_tesseract = "C://Users//willi//AppData//Local//Programs//Tesseract-OCR//tesseract.exe"
# providing the tesseract executable location
pytesseract.tesseract_cmd = path_to_tesseract

image_path = "C://Users//willi//Desktop//Ad skrnshts//ad6busy.png"
image = cv2.imread(image_path)
modified_image = np.zeros(image.shape, dtype=np.uint8)

triangle_vertices_list = triangle_detector.find_triangle_vertices(image_path)
for triangle_vertices in triangle_vertices_list: 
    top_left_vertex, bottom_left_vertex, far_right_vertex = triangle_vertices
    # top left (start) and bottom right (end) corners of the rectangle that presumably encloses the "skip ad" text
    rectangle_start_x = top_left_vertex[0] - abs(far_right_vertex[0] - top_left_vertex[0]) * 20
    rectangle_start_y = top_left_vertex[1] - abs(bottom_left_vertex[1] - top_left_vertex[1])
    rectangle_end_x = top_left_vertex[0]
    rectangle_end_y = bottom_left_vertex[1] + abs(bottom_left_vertex[1] - top_left_vertex[1])
    rectangle_start = (max(0, rectangle_start_x), max(0, rectangle_start_y))
    rectangle_end = (rectangle_end_x, min(image.shape[0] - 1, rectangle_end_y))
    modified_image = cv2.rectangle(modified_image, rectangle_start, rectangle_end, (255, 255, 255), -1)

# in the modified_image, all parts of the image are turned to black 
# except the areas that are to the left of a "right arrow" triangle
coors_to_preserve = np.where(modified_image == (255, 255, 255))[0], np.where(modified_image == (255, 255, 255))[1] 
modified_image[coors_to_preserve] = image[coors_to_preserve]
gray_modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
_, text_threshold = cv2.threshold(gray_modified_image, 128, 255, cv2.THRESH_BINARY) 

cv2.imshow("Text Threshold", cv2.resize(text_threshold, (0,0), fx = 0.7, fy = 0.7))
cv2.waitKey(0)
cv2.destroyAllWindows()

pil_modified_image = Image.fromarray(text_threshold)

# image to string of predicted text
result = pytesseract.image_to_data(pil_modified_image) # image_to_data gives more info (than image_to_string)
result = result.strip()

####### *** CORRECTION TO THE COMMENT IN PREVIOUS COMMIT: error caused by row with 11 elements instead of 12 (the text element was omitted) *** #######
# turning tab-delimited result string to pandas dataframe
# the first row of result is the column headers (which we ignore in the body of the dataframe by doing result.split("\n")[1:])
result_list = []
for row in result.split("\n")[1:]:
    row_list = row.split("\t")
    if len(row_list) == 12:
        result_list.append(row_list)
result_dataframe = pd.DataFrame(result_list, columns=['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
                                    'left', 'top', 'width', 'height', 'conf', 'text'])

#print (result_dataframe)

for i in range(len(result_dataframe)):
    row = result_dataframe.iloc[i]
    # if confidence > 59% and the text is not None, not just white space and has more than one character, then...
    if float(row['conf']) > 59 and row['text'] != None and len(row['text'].strip()) > 1:
        start_point = (int(row['left']), int(row['top']))
        end_point = (int(row['left']) + int(row['width']), int(row['top']) + int(row['height']))
        image = cv2.rectangle(image, start_point, end_point, (0, 0, 255), 1)
        print(row['text'])

factor = 0.7
display_image = cv2.resize(image, (0,0), fx = factor, fy = factor)

cv2.imshow("Image", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()