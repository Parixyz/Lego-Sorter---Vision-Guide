import cv2
import os
from ProcessingTools.Classical.filters import ImageFilters
import numpy as np
def main():
    # Load the test image
    image_path = "Test.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found at {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Create output directory
    output_dir = "output/filters"
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate the ImageFilters class
    image_filters = ImageFilters()

    # List of filters to apply
    filters_to_apply = [
        ("Gaussian Blur", image_filters.gaussian_blur, {"sigma": 2}),
        ("Uniform Filter", image_filters.uniform_filter, {"size": 3}),
        ("Sobel Edges", image_filters.sobel_edges, {}),
        ("Laplacian Filter", image_filters.laplacian_filter, {}),
        ("Canny Edges", image_filters.canny_edges, {"threshold1": 50, "threshold2": 150}),
        ("Bilateral Filter", image_filters.bilateral_filter, {"d": 9, "sigma_color": 75, "sigma_space": 75}),
        ("Median Blur", image_filters.median_blur, {"ksize": 5}),
        ("Histogram Equalization", image_filters.histogram_equalization, {}),
        ("Adaptive Histogram Equalization", image_filters.adaptive_histogram_equalization, {"clip_limit": 2.0, "tile_grid_size": (8, 8)}),
        ("Nearest Neighbor Interpolation", image_filters.nearest_neighbor_interpolation, {"scale_factor": 1.5}),
        ("K-Means Segmentation", image_filters.kmeans_segmentation, {"k": 3}),
    ]

    # Apply all filters and save results
    for name, method, params in filters_to_apply:
        try:
            result = method(image, **params) if params else method(image)
            if isinstance(result, np.ndarray):
                output_path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.png")
                cv2.imwrite(output_path, result)
                print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to apply {name}: {e}")

    # Perform Color Segmentation
    try:
        red_segment, blue_segment, green_segment = image_filters.color_segmentation(image)
        cv2.imwrite(os.path.join(output_dir, "red_segment.png"), red_segment)
        cv2.imwrite(os.path.join(output_dir, "blue_segment.png"), blue_segment)
        cv2.imwrite(os.path.join(output_dir, "green_segment.png"), green_segment)
        print("Performed Color Segmentation")
    except Exception as e:
        print(f"Failed to perform Color Segmentation: {e}")

    # Perform Histogram Analysis
    try:
        histogram_output_path = os.path.join(output_dir, "histogram_plot.png")
        image_filters.histogram_analysis(image, output_path=histogram_output_path)
        print("Performed Histogram Analysis")
    except Exception as e:
        print(f"Failed to perform Histogram Analysis: {e}")

if __name__ == "__main__":
    main()
