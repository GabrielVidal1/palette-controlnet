from math import ceil
from typing import Callable
from sklearn.cluster import KMeans

import numpy as np
from palette_to_image import image_from_palette


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
