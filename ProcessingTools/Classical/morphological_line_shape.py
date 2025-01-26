import cv2
import numpy as np

class MorphologicalLineShape:
    def __init__(self):
        pass

    # --- Morphological Operations ---
    def morphological_opening(self, image, kernel_size=3):
        """Apply morphological opening to remove noise."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def morphological_closing(self, image, kernel_size=3):
        """Apply morphological closing to fill small holes."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def morphological_gradient(self, image, kernel_size=3):
        """Apply morphological gradient to highlight edges."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    def dilation(self, image, kernel_size=3):
        """Apply morphological dilation to expand objects in the image."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.dilate(image, kernel, iterations=1)

    def erosion(self, image, kernel_size=3):
        """Apply morphological erosion to thin objects in the image."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.erode(image, kernel, iterations=1)

    # --- Line Detection ---
    def hough_lines(self, image, rho=1, theta=np.pi/180, threshold=100):
        """Detect lines using the Hough Transform."""
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image

    # --- Shape Detection ---
    def polygon_approximation(self, image, epsilon_factor=0.02):
        """Approximate contours with polygons."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approx_image = image.copy()
        for contour in contours:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(approx_image, [approx], -1, (0, 255, 0), 2)
        return approx_image

    def corner_detection(self, image, block_size=2, ksize=3, k=0.04):
        """Detect corners in the image using the Harris Corner Detector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        corners = cv2.cornerHarris(gray, block_size, ksize, k)
        corners = cv2.dilate(corners, None)
        corner_image = image.copy()
        corner_image[corners > 0.01 * corners.max()] = [0, 0, 255]
        return corner_image

    # --- Object Segmentation Pipeline ---
    def segment_objects(self, image, k=3, blur_method="gaussian", blur_kernel=5):
        """
        Segments objects in the image using filters and K-Means.
        Args:
            image: Input image (BGR format).
            k: Number of clusters for K-Means.
            blur_method: Blurring method ('gaussian' or 'median').
            blur_kernel: Kernel size for blurring.
        Returns:
            List of segmented object images.
        """
        # Step 1: Apply Blurring
        if blur_method == "gaussian":
            blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        elif blur_method == "median":
            blurred = cv2.medianBlur(image, blur_kernel)
        else:
            blurred = image

        # Step 2: Apply K-Means Clustering
        reshaped = blurred.reshape((-1, 3))
        reshaped = np.float32(reshaped)
        _, labels, centers = cv2.kmeans(
            reshaped,
            k,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        segmented = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)

        # Step 3: Extract Masks for Each Cluster
        hsv_segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
        object_masks = []
        for i, center in enumerate(centers):
            color_hsv = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_BGR2HSV)[0][0]
            lower_bound = np.array([max(0, color_hsv[0] - 10), 50, 50])
            upper_bound = np.array([min(180, color_hsv[0] + 10), 255, 255])
            mask = cv2.inRange(hsv_segmented, lower_bound, upper_bound)

            # Step 4: Refine the Mask
            mask = self.morphological_opening(mask, kernel_size=5)
            mask = self.morphological_closing(mask, kernel_size=5)
            object_masks.append(mask)

        # Step 5: Extract Objects Using Masks
        segmented_objects = []
        for mask in object_masks:
            segmented_object = cv2.bitwise_and(image, image, mask=mask)
            segmented_objects.append(segmented_object)

        return segmented_objects
