import cv2
import numpy as np

class Collage:
    def __init__(self, org, size_x, size_y, image_list):
        """
        Initialize the Collage class.
        :param org: Original image (used as base canvas).
        :param size_x: Width of the collage.
        :param size_y: Height of the collage.
        :param image_list: List of images or nested lists of images.
        """
        self.org = org
        self.size_x = size_x
        self.size_y = size_y
        self.image_list = image_list

    def create_collage(self):
        """
        Create a collage based on the given image list.
        :return: Final collage image.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        status = self.image_list[0]  # First element determines the layout ('v' or 'h')
        divisions = len(self.image_list) - 1  # Number of images to process

        # Create a blank canvas
        collage = np.zeros((self.size_y, self.size_x, 3), dtype=np.uint8)

        for i in range(1, len(self.image_list)):
            if status == 'v':
                # Vertical division
                sub_width = self.size_x // divisions
                sub_height = self.size_y

                img = self.image_list[i]
                if isinstance(img, list):
                    # If it's a nested list, recursively create a collage
                    sub_collage = Collage(collage, sub_width, sub_height, img).create_collage()
                    collage[:, (i - 1) * sub_width:i * sub_width] = cv2.resize(
                        sub_collage, (sub_width, sub_height))
                elif isinstance(img, tuple):
                    # Handle tuple: add text from the tuple
                    img, text = img
                    labeled_img = img.copy()
                    cv2.putText(labeled_img, text, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    collage[:, (i - 1) * sub_width:i * sub_width] = cv2.resize(labeled_img, (sub_width, sub_height))
                elif isinstance(img, np.ndarray):
                    # Add the image to the correct section
                    collage[:, (i - 1) * sub_width:i * sub_width] = cv2.resize(img, (sub_width, sub_height))

            else:
                # Horizontal division
                sub_width = self.size_x
                sub_height = self.size_y // divisions

                img = self.image_list[i]
                if isinstance(img, list):
                    # If it's a nested list, recursively create a collage
                    sub_collage = Collage(collage, sub_width, sub_height, img).create_collage()
                    collage[(i - 1) * sub_height:i * sub_height, :] = cv2.resize(
                        sub_collage, (sub_width, sub_height))
                elif isinstance(img, tuple):
                    # Handle tuple: add text from the tuple
                    img, text = img
                    labeled_img = img.copy()
                    cv2.putText(labeled_img, text, (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    collage[(i - 1) * sub_height:i * sub_height, :] = cv2.resize(labeled_img, (sub_width, sub_height))
                elif isinstance(img, np.ndarray):
                    # Add the image to the correct section
                    collage[(i - 1) * sub_height:i * sub_height, :] = cv2.resize(img, (sub_width, sub_height))

        return collage
