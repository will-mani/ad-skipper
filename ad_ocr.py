from PIL import Image
from pytesseract import pytesseract
import cv2 
import pandas as pd
import time

path_to_tesseract = "C://Users//willi//AppData//Local//Programs//Tesseract-OCR//tesseract.exe"
# Providing the tesseract executable location
pytesseract.tesseract_cmd = path_to_tesseract

images = ["ad3window.png", "ad4FULL.png", "ad5BRIGHT.png", "ad5DIM.png", "ad5mini.png", "ad6busy.png", "ad8DARK.png"]

for image in images:
    image_path = "Ad skrnshts//" + image

    start_time = time.time()
    # Image to string of predicted text
    result = pytesseract.image_to_data(Image.open(image_path)) # image_to_data gives more info (than image_to_string)
    result = result.strip()

    print(image)
    end_time = time.time()
    print(end_time - start_time, "seconds")
    print()

    # Turning tab-delimited result string to pandas dataframe
    result_dataframe = pd.DataFrame([x.split('\t') for x in result.split("\n")[1:]],
                             columns=['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
                                      'left', 'top', 'width', 'height', 'conf', 'text'])

    #print (result_dataframe)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for i in range(len(result_dataframe)):
        row = result_dataframe.iloc[i]

        # If confidence > 59%, the text is not None and not just white space, then...
        if float(row['conf']) > 59 and row['text'] != None and len(row['text'].strip()) > 1:
            
            start_point = (int(row['left']), int(row['top']))
            end_point = (int(row['left']) + int(row['width']), int(row['top']) + int(row['height']))
            image = cv2.rectangle(image, start_point, end_point, (0, 0, 255), 1)

    factor = 0.8
    display_image = cv2.resize(image, (0,0), fx= factor, fy = factor)

    cv2.imshow("Image", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
