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

def detect_next_contours(screen_thresh):
    next_contours_list = []
    screen_contours, _ = cv2.findContours(screen_thresh, 2, 1)
    for curr_contour in screen_contours:
        difference = cv2.matchShapes(model_triangle_contour, curr_contour, 1, 0.0)
        if difference < 0.048:
            next_contours_list.append(curr_contour)
    return next_contours_list


def detect_skip_text(screen_image, next_contours_list):
    skip_box_center = None

    for contour in next_contours_list:
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
            if len(row['text']) > 2 and row['text'].lower() in 'skip ads':
                print(row['text'], row['confidence'])
                bbox_array = np.array(row['bbox'])
                center_x = int(np.mean(bbox_array[:, 0]) + left_most_point)
                center_y = int(np.mean(bbox_array[:, 1]) + top_point)
                skip_box_center = [center_x, center_y]
                return skip_box_center
            
    return skip_box_center


# test_image = cv2.imread('test_images/window.png')
# center = detect_skip_text(test_image)
# print(center)

def capture_screen(image_grab_bbox, image_ratio=1):
    screenshot = ImageGrab.grab(bbox=image_grab_bbox)
    bgr_screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    resized_screenshot = cv2.resize(bgr_screenshot, (0, 0), fx=image_ratio, fy=image_ratio)
    return resized_screenshot

screen_ratio = 0.7
whole_screen = capture_screen(image_grab_bbox=None, image_ratio=screen_ratio)
# Select region of interest then press ENTER
screen_ratio_roi = cv2.selectROI("ROI", whole_screen) # [top_x, top_y, width, height]
roi = np.floor(np.array(screen_ratio_roi) * (1 / screen_ratio)).tolist()


while True:
    try:
        roi_screenshot = capture_screen(image_grab_bbox=(int(roi[0]), int(roi[1]), int(roi[0] + roi[2]), int(roi[1] + roi[3])))
    except:
        continue
    
    show_ratio = 0.3
    cv2.imshow("ROI", cv2.resize(roi_screenshot, (0, 0), fx=show_ratio, fy=show_ratio))

    screen_grayscale = cv2.cvtColor(roi_screenshot, cv2.COLOR_BGR2GRAY)
    _, screen_thresh = cv2.threshold(screen_grayscale, 200, 255,0)
    next_contours_list = detect_next_contours(screen_thresh)

    bgr_screen_thresh = cv2.cvtColor(screen_thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(bgr_screen_thresh, next_contours_list, -1, (0, 255, 0), -1)
    cv2.imshow("Thresh & Tris", cv2.resize(bgr_screen_thresh, (0, 0), fx=show_ratio, fy=show_ratio))
    
    skip_center = detect_skip_text(roi_screenshot, next_contours_list)

    if skip_center != None:
        x = skip_center[0] + roi[0]
        y = skip_center[1] + roi[1]
        if (x > 0 and x < pyautogui.size()[0]) and (y > 0 and y < pyautogui.size()[1]):
            print("Skip at x =", x, ", y =", y, '\n')
            pyautogui.moveTo(x, y, 0.5, pyautogui.easeInQuad) # x, y, 0.5 secs, start slow then end fast
            pyautogui.click()
            # pyautogui.click(x, y)

            # 1 second pause
            start_time_stamp = time.time()
            while True:
                if time.time() - start_time_stamp > 1:
                    break

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
