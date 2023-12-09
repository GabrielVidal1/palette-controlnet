from PIL import Image
import colorsys
import numpy as np


def rotate_hue(image, hue_rotation):
    """
    Rotate the hue of an image by a certain number of degrees.

    Parameters:
    image (PIL.Image): The input PIL image to modify.
    degrees (float): The number of degrees to rotate the hue.

    Returns:
    PIL.Image: The modified image with the hue rotated.
    """

    # Convert the image to HSV
    hsv_image = image.convert("HSV")
    hsv_data = np.array(hsv_image)

    # Calculate the hue rotation
    hue_rotation = int(hue_rotation * 255)

    # Rotate the hue channel (0)
    hsv_data[..., 0] = (hsv_data[..., 0] + hue_rotation) % 256

    # Convert the HSV data back to an image and then convert it to RGB
    new_image = Image.fromarray(hsv_data, "HSV").convert("RGB")

    return new_image
