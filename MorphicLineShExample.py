from ProcessingTools.Classical.filters import ImageFilters
from ProcessingTools.Classical.morphological_line_shape import MorphologicalLineShape
import os
import cv2

def main():
    image_path = "Test.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found at {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    output_dir = "output/filters"
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate the classes
    filters = ImageFilters()
    morph_shapes = MorphologicalLineShape()

    # Apply morphological gradient
    result = morph_shapes.morphological_gradient(image)
    cv2.imwrite(os.path.join(output_dir, "morphological_gradient.png"), result)

    # Line detection
    result = morph_shapes.hough_lines(image)
    cv2.imwrite(os.path.join(output_dir, "hough_lines.png"), result)

    # Shape detection
    result = morph_shapes.polygon_approximation(image)
    cv2.imwrite(os.path.join(output_dir, "polygon_approximation.png"), result)

    # Corner detection
    result = morph_shapes.corner_detection(image)
    cv2.imwrite(os.path.join(output_dir, "corner_detection.png"), result)

    print("All operations applied successfully.")

if __name__ == "__main__":
    main()
