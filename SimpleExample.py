import cv2
import os
import numpy as np
from ProcessingTools.Classical.morphological_line_shape import MorphologicalLineShape

def main():
    # Load the test image
    image_path = "Test.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found at {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    # Create output directory
    output_dir = "output/dominant_colors"
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate the MorphologicalLineShape class
    morph_shapes = MorphologicalLineShape()

    # Step 1: Segment objects and extract dominant colors
    segmented_objects = morph_shapes.segment_objects(image, k=4, blur_method="gaussian", blur_kernel=5)

    # Step 2: Save each object's image and extract its dominant color
    dominant_colors = []
    for i, obj in enumerate(segmented_objects):
        # Save the segmented object
        obj_path = os.path.join(output_dir, f"object_{i}.png")
        cv2.imwrite(obj_path, obj)

        # Compute the dominant color
        reshaped = obj.reshape((-1, 3))
        reshaped = reshaped[~np.all(reshaped == 0, axis=1)]  # Exclude black pixels (background)
        if reshaped.size > 0:
            mean_color = np.mean(reshaped, axis=0).astype(np.uint8)  # Average color
            dominant_colors.append({
                "Object": i,
                "Dominant Color (BGR)": mean_color.tolist()
            })

    # Print and save the dominant colors
    print("Dominant Colors Detected:")
    for color_info in dominant_colors:
        print(color_info)

    # Save dominant colors as an image
    color_preview = np.zeros((100, len(dominant_colors) * 100, 3), dtype=np.uint8)
    for i, color_info in enumerate(dominant_colors):
        color = color_info["Dominant Color (BGR)"]
        color_preview[:, i * 100:(i + 1) * 100] = color
    cv2.imwrite(os.path.join(output_dir, "dominant_colors_preview.png"), color_preview)

    print("Dominant colors saved successfully.")

if __name__ == "__main__":
    main()
