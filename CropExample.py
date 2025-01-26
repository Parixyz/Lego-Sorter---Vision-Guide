import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, uniform_filter, variance
from scipy.signal import convolve2d
from skimage.filters import unsharp_mask, threshold_otsu, rank
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import disk

class Filters:
    @staticmethod
    def gaussian_blur(image, sigma=1):
        """Apply Gaussian Blur to the image."""
        return gaussian_filter(image, sigma=sigma)

    @staticmethod
    def uniform_filter(image, size=3):
        """Apply a uniform (mean) filter."""
        return uniform_filter(image, size=size)

    @staticmethod
    def variance_filter(image, size=3):
        """Apply a variance filter."""
        return variance(image, size=size)

    @staticmethod
    def sobel_edges(image):
        """Detect edges using the Sobel filter."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        dx = sobel(gray, axis=0)
        dy = sobel(gray, axis=1)
        return np.hypot(dx, dy)

    @staticmethod
    def threshold_otsu(image):
        """Apply Otsu's thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        thresh = threshold_otsu(gray)
        binary = gray > thresh
        return binary

    @staticmethod
    def tv_denoise(image, weight=0.1):
        """Denoise using Total Variation (TV) regularization."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return denoise_tv_chambolle(gray, weight=weight, multichannel=False)

    @staticmethod
    def laplacian_filter(image):
        """Apply Laplacian filter for edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F)

    @staticmethod
    def canny_edges(image, threshold1=100, threshold2=200):
        """Detect edges using the Canny edge detector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Canny(gray, threshold1, threshold2)

    @staticmethod
    def adaptive_threshold(image):
        """Apply adaptive thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def hough_lines(image, rho=1, theta=np.pi/180, threshold=150):
        """Detect lines using Hough Line Transform."""
        edges = cv2.Canny(image, 50, 150)  # Detect edges
        return cv2.HoughLines(edges, rho, theta, threshold)

    @staticmethod
    def hough_lines_probabilistic(image, rho=1, theta=np.pi/180, threshold=80, min_line_length=50, max_line_gap=10):
        """Detect lines using Probabilistic Hough Line Transform."""
        edges = cv2.Canny(image, 50, 150)  # Detect edges
        return cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    @staticmethod
    def find_contours(image):
        """Find contours in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def draw_contours(image, contours):
        """Draw contours on the image."""
        return cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    @staticmethod
    def color_mask(image, lower_bound, upper_bound):
        """Apply a mask to detect specific colors in the image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return cv2.bitwise_and(image, image, mask=mask)

    @staticmethod
    def morphological_opening(image, kernel_size=3):
        """Apply morphological opening to remove noise."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def morphological_closing(image, kernel_size=3):
        """Apply morphological closing to fill small holes."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def morphological_gradient(image, kernel_size=3):
        """Apply morphological gradient to highlight edges."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Failed to load the image. Check the path.")

    filters = Filters()

    # Example of color masking
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    red_mask = filters.color_mask(image, lower_red, upper_red)
    cv2.imshow("Red Mask", red_mask)

    # Example of Hough line detection
    lines = filters.hough_lines(image)
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
    cv2.imshow("Hough Lines", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
