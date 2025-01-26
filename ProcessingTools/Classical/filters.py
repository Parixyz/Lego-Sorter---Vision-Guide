import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from skimage.filters import unsharp_mask, threshold_otsu, rank
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import disk
from sklearn.cluster import KMeans

class ImageFilters:
    def __init__(self):
        pass  # Constructor for potential future use

    def gaussian_blur(self, image, sigma=1):
        """Apply Gaussian Blur to the image."""
        return gaussian_filter(image, sigma=sigma)

    def uniform_filter(self, image, size=3):
        """Apply a uniform (mean) filter."""
        return uniform_filter(image, size=size)

    def sobel_edges(self, image):
        """Detect edges using the Sobel filter."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        dx = sobel(gray, axis=0)
        dy = sobel(gray, axis=1)
        return np.hypot(dx, dy)

    def laplacian_filter(self, image):
        """Apply Laplacian filter for edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F)

    def canny_edges(self, image, threshold1=100, threshold2=200):
        """Detect edges using the Canny edge detector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Canny(gray, threshold1, threshold2)

    def bilateral_filter(self, image, d=9, sigma_color=75, sigma_space=75):
        """Apply Bilateral Filter for edge-preserving smoothing."""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def median_blur(self, image, ksize=5):
        """Apply Median Blur for noise reduction."""
        return cv2.medianBlur(image, ksize)

    def histogram_equalization(self, image):
        """Apply histogram equalization."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.equalizeHist(gray)

    def adaptive_histogram_equalization(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply Adaptive Histogram Equalization."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    def nearest_neighbor_interpolation(self, image, scale_factor=2):
        """Resize an image using nearest neighbor interpolation."""
        height, width = image.shape[:2]
        new_size = (int(width * scale_factor), int(height * scale_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

    def edge_count(self, image):
        """Count the number of edges in an image using the Sobel operator."""
        edges = self.sobel_edges(image)
        return np.sum(edges > 0)

    def histogram_analysis(self, image, output_path=None):
        """Compute the histogram of an image and optionally save the plot."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        if output_path:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.title("Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.plot(hist)
            plt.xlim([0, 256])
            plt.grid()
            plt.savefig(output_path)
            plt.close()
            print(f"Saved histogram plot: {output_path}")

        return hist

    def color_segmentation(self, image):
        """Segment the image into different colors based on filters."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Broadened red range
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 | mask_red2
        red_segment = cv2.bitwise_and(image, image, mask=mask_red)

        # Broadened blue range
        lower_blue = np.array([90, 100, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_segment = cv2.bitwise_and(image, image, mask=mask_blue)

        # Broadened green range
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        green_segment = cv2.bitwise_and(image, image, mask=mask_green)

        return red_segment, blue_segment, green_segment

    def kmeans_segmentation(self, image, k=3):
        """Apply K-Means clustering for image segmentation."""
        reshaped = image.reshape((-1, 3))
        reshaped = np.float32(reshaped)
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(reshaped)
        centers = np.uint8(kmeans.cluster_centers_)
        segmented_image = centers[labels.flatten()].reshape(image.shape)
        return segmented_image
