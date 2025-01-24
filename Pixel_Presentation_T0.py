# Pixel Representation and Real-Time Analysis

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Callback function to display pixel information
def show_pixel_info(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Extract pixel values
        b, g, r = image[y, x]

        # Create images with individual channels
        red_channel = np.zeros_like(image)
        red_channel[:, :, 2] = r

        green_channel = np.zeros_like(image)
        green_channel[:, :, 1] = g

        blue_channel = np.zeros_like(image)
        blue_channel[:, :, 0] = b

        # Convert RGB to HSI for the pixel
        hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, i = hsi_image[y, x]

        # Display information
        print(f"Pixel at ({x}, {y}):")
        print(f"  RGB: R={r}, G={g}, B={b}")
        print(f"  HSI: H={h}, S={s}, I={i}")

        # Show individual channel images
        cv2.imshow("Red Channel", red_channel)
        cv2.imshow("Green Channel", green_channel)
        cv2.imshow("Blue Channel", blue_channel)

# Load an image
image_path = cv2.samples.findFile("sudoku.png")  # Replace with your image path
image = cv2.imread(image_path)

# Display the image and set the callback function
cv2.imshow("Original Image", image)
cv2.setMouseCallback("Original Image", show_pixel_info)

# Wait until the user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()

# Purpose of this code:
# - This program allows you to interact with an image and get real-time pixel information.
# - It separates channels (Red, Green, Blue) and shows how the pixel values correspond to these channels.
# - It converts RGB to HSI and provides the Hue, Saturation, and Intensity values of the pixel.
#
# Key points explained:
# - `cv2.EVENT_MOUSEMOVE`: Captures mouse movement to provide real-time pixel analysis.
# - `cv2.split`: Not used here directly but conceptually shows how channels can be isolated.
# - `cv2.cvtColor`: Converts the image to a different color space (e.g., HSV for HSI representation).
#
# You can extend this by adding support for other color spaces or saving the visualizations for later analysis.
