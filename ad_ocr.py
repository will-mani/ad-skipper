import cv2
import numpy as np
from PIL import ImageGrab
import time
import easyocr
import pandas as pd
import pyautogui

ocr_reader = easyocr.Reader(['en'], gpu=False)

model_next_image = cv2.imread('next.png', cv2.IMREAD_GRAYSCALE)
_, next_thresh = cv2.threshold(model_next_image, 127, 255,0)
model_next_contours, _ = cv2.findContours(next_thresh, 2, 1)
model_triangle_contour = model_next_contours[1]

def detect_next_contours(screen_image):
    next_contours_list = []
    screen_grayscale = cv2.cvtColor(screen_image, cv2.COLOR_BGR2GRAY)
    _, screen_thresh = cv2.threshold(screen_grayscale, 200, 255,0)
    screen_contours, _ = cv2.findContours(screen_thresh, 2, 1)
    for curr_contour in screen_contours:
        difference = cv2.matchShapes(model_triangle_contour, curr_contour, 1, 0.0)
        if difference < 0.125:
            next_contours_list.append(curr_contour)
            print(difference)
    return next_contours_list


def detect_skip_text(screen_image):
    skip_box_center = None

    next_contours_list = detect_next_contours(screen_image)
    for contour in next_contours_list:
        cv2.drawContours(roi_screenshot, [contour], 0, (255, 0, 0), -1)
        cv2.drawContours(roi_screenshot, [contour], 0, (0, 0, 255), 1)
        reshaped_contour = np.reshape(contour, (contour.shape[0], contour.shape[2]))

        min_y = min(reshaped_contour[:, 1])
        max_y = max(reshaped_contour[:, 1])

        height = abs(max_y - min_y)
        top_point = max(0, min_y - (height * 2))
        bottom_point = min(screen_image.shape[0] - 1, max_y + (height * 2))

        min_x = min(reshaped_contour[:, 0])
        max_x = max(reshaped_contour[:, 0])

        width = abs(max_x - min_x)
        left_most_point = max(0, min_x - (width * 16))
        right_most_point = min(screen_image.shape[1] - 1, max_x + width)

        cropped_image = screen_image[top_point:bottom_point, left_most_point:right_most_point]

        ocr_results = ocr_reader.readtext(cropped_image)
        results_df = pd.DataFrame(data=ocr_results, columns=['bbox', 'text', 'confidence'])

        for i in range(len(results_df)):
            row = results_df.iloc[i]
            if len(row['text']) > 2 and row['text'].lower() in 'skip ad':
                print(row['text'])
                bbox_array = np.array(row['bbox'])
                center_x = int(np.mean(bbox_array[:, 0]) + left_most_point)
                center_y = int(np.mean(bbox_array[:, 1]) + top_point)
                skip_box_center = [center_x, center_y]
                return skip_box_center
            
    return skip_box_center


# test_image = cv2.imread('test_images/window.png')
# center = detect_skip_text(test_image)
# print(center)

def capture_screen(image_grab_bbox):
    screenshot = ImageGrab.grab(bbox=image_grab_bbox)
    bgr_screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    resized_screenshot = cv2.resize(bgr_screenshot, (0, 0), fx = 1, fy = 1)
    return resized_screenshot

whole_screen = capture_screen(image_grab_bbox=None)
# Select region of interest then press ENTER
roi = cv2.selectROI("ROI", whole_screen) # [top_x, top_y, width, height]


while True:
    roi_screenshot = capture_screen(image_grab_bbox=(int(roi[0]), int(roi[1]), int(roi[0]+roi[2]), int(roi[1]+roi[3])))
    
    skip_center = detect_skip_text(roi_screenshot)

    if skip_center != None:
        x = skip_center[0] + roi[0]
        y = skip_center[1] + roi[1]
        print("Skip at x =", x, ", y =", y)
        pyautogui.moveTo(x, y, 2, pyautogui.easeInQuad) # x, y, 2 secs, start slow then end fast
        pyautogui.click()
        # pyautogui.click(x, y)

    cv2.imshow("ROI", roi_screenshot)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
        
