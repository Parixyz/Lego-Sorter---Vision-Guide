from visualization_tools.CollageMaker import Collage
import cv2
import numpy as np

if __name__ == "__main__":
    # Open camera for real-time capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Could not open the camera.")

    while True:
        ret, img_original = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        # Generate RGB channels
        b, g, r = cv2.split(img_original)
        img_r = (cv2.merge([r, np.zeros_like(r), np.zeros_like(r)]), "Red Channel")
        img_g = (cv2.merge([np.zeros_like(g), g, np.zeros_like(g)]), "Green Channel")
        img_b = (cv2.merge([np.zeros_like(b), np.zeros_like(b), b]), "Blue Channel")

        # Convert to HSI (HSV in OpenCV)
        hsi_image = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
        h, s, i = cv2.split(hsi_image)
        img_h = (cv2.applyColorMap(h, cv2.COLORMAP_JET), "Hue")
        img_s = (cv2.applyColorMap(s, cv2.COLORMAP_JET), "Saturation")
        img_i = (cv2.applyColorMap(i, cv2.COLORMAP_JET), "Intensity")

        # Create nested image list
        image_list = [
            'v',
            (img_original, "Original Image"),  # Original image
            [
                'h',
                [
                    'h', img_r, img_g, img_b  # RGB channels
                ],
                [
                    'h', img_h, img_s, img_i  # HSI components
                ]
            ]
        ]

        # Initialize collage
        size_x, size_y = 800, 600
        org = np.zeros((size_y, size_x, 3), dtype=np.uint8)
        collage = Collage(org, size_x, size_y, image_list).create_collage()

        # Display the collage
        cv2.imshow("Real-Time Collage", collage)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
