from math import ceil
from typing import Callable
from PIL import Image

import matplotlib
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def extract_colors(image, num_colors=6):
    """
    Extracts the top n dominant colors from the image in the given path
    and return them as a list of RGB tuples
    """
    # Load image
    image = image.resize((150, 150))  # Resize for faster processing
    image_data = np.array(image)
    image_data = image_data.reshape((-1, 3))

    # Extract the top n dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image_data)

    # Get the colors and sort them by the frequency
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    counts = np.bincount(labels)
    colors = colors[np.argsort(counts)[::-1]]
    print("colors", colors)
    return [tuple(colors[i].astype(int)) for i in range(len(colors))]


def image_from_palette(main_colors, size=(150, 150), square_by_side=4):
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


class Palettify(Callable):
    def __init__(self, num_colors=6):
        self.num_colors = num_colors

    def __call__(self, img):
        w, h = img.size
        colors = extract_colors(img, self.num_colors)

        return image_from_palette(colors, size=(w, h))

    def __repr__(self):
        return "custom augmentation"


def hex_to_color(hex):
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


def hex_string_to_image(hex_string, size=(150, 150)):
    """
    for example
    acbdba,cddddd,a599b5,2e2f2f,051014
    """
    colors = [hex_to_color(hex) for hex in hex_string.split(",")]
    return image_from_palette(colors, size=size)
