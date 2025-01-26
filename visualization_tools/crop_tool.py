import cv2
import numpy as np

class CropTool:
    def __init__(self):
        """
        Initialize the CropTool class for handling static and interactive image cropping.
        """
        pass

    @staticmethod
    def crop_static(image, top_left, bottom_right):
        """
        Crops a region of interest (ROI) from the image based on given coordinates.
        :param image: The input image to crop.
        :param top_left: Tuple (x, y) representing the top-left corner of the crop.
        :param bottom_right: Tuple (x, y) representing the bottom-right corner of the crop.
        :return: The cropped image.
        """
        x1, y1 = top_left
        x2, y2 = bottom_right
        return image[int(y1):int(y2), int(x1):int(x2)]

    @staticmethod
    def crop_interactive(image):
        """
        Opens an interactive window to crop a region of interest (ROI) from the image.
        :param image: The input image to crop.
        :return: The cropped image.
        """
        roi = cv2.selectROI("Interactive Crop", image, fromCenter=False, showCrosshair=True)
        if roi == (0, 0, 0, 0):
            print("No region selected.")
            return None
        x, y, w, h = roi
        cropped = image[int(y):int(y + h), int(x):int(x + w)]
        cv2.destroyWindow("Interactive Crop")
        return cropped