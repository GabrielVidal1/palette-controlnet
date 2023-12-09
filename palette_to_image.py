from math import ceil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def debug_palette(main_colors, size=(150, 150)):
    """
    Create an image that displays the palette of colors provided.
    """
    img = Image.new("RGB", size)
    width = int(size[0] / len(main_colors))
    for i, color in enumerate(main_colors):
        img.paste(
            color,
            (
                int(i * width),
                0,
                int((i + 1) * width),
                size[1],
            ),
        )

    return img


def image_from_palette_squares(main_colors, size=(150, 150), square_by_side=4):
    squares = [i**2 for i in range(0, 8)]
    if i := squares.index(len(main_colors)) == -1:
        nearest_square = ceil(len(main_colors) ** 0.5)
        print(
            f"warning: number of colors is not a square number, rounding to {nearest_square}"
        )
        squares = squares[:nearest_square]
    i = squares.index(len(main_colors))

    square_side = int(size[0] / square_by_side)

    # Create a square with all colors
    square_img = Image.new("RGB", (square_side, square_side))
    small_square_side = int(square_side / i) + 1
    for x in range(i):
        for y in range(i):
            # Use the formula to determine the color index
            color_index = (x + i * y) % len(main_colors)
            # Get the color from the main colors
            color = tuple(main_colors[color_index])
            # Assign the color to the pixel
            square_img.paste(
                color,
                (
                    int(x * small_square_side),
                    int(y * small_square_side),
                    int((x + 1) * small_square_side),
                    int((y + 1) * small_square_side),
                ),
            )

    new_image = Image.new("RGB", size)

    # Assign colors to each pixel based on the provided formula
    for x in range(square_by_side):
        for y in range(square_by_side):
            new_image.paste(
                square_img,
                (
                    int(x * square_side),
                    int(y * square_side),
                    int((x + 1) * square_side),
                    int((y + 1) * square_side),
                ),
            )
            # Rotate square 90 degrees
            square_img = square_img.rotate(90 * (x + y + 1) % 360)

    return new_image


def image_from_palette(main_colors, size=(150, 150)):
    """
    Create an image that displays the palette of colors provided.
    """
    # Number of colors
    num_colors = len(main_colors)

    # Create a white noise image (random values between 0 and 1)
    noise = np.random.rand(size[0], size[1])

    # Scale and round the values to nearest index
    scaled_indices = np.round(noise * (num_colors - 1)).astype(int)

    # Create an empty image
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Assign colors to pixels based on the rounded indices
    for i in range(num_colors):
        image[scaled_indices == i] = main_colors[i]

    # Convert to an image and return
    return Image.fromarray(image, mode="RGB")
