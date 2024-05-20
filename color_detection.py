import sys
import json
from collections import Counter

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2lab


def rgb2hex(color):
    return '#{:02x}{:02x}{:02x}'.format(
        int(color[0]), int(color[1]), int(color[2])
    )


def hex2name(hex_color):
    with open('color_tables/NBS-ISCC-rgb.json', 'r') as f:
        color_table = json.loads(f.read())

    hex_rgb_colors = list(color_table.keys())

    r = [int(hex[1:3], 16) for hex in hex_rgb_colors]  # List of red elements.
    g = [int(hex[3:5], 16) for hex in hex_rgb_colors]  # List of green elements.
    b = [int(hex[5:7], 16) for hex in hex_rgb_colors]  # List of blue elements.

    r = np.asarray(r, np.uint8)  # Convert r from list to array (of uint8 elements)
    g = np.asarray(g, np.uint8)  # Convert g from list to array
    b = np.asarray(b, np.uint8)  # Convert b from list to array

    rgb = np.dstack((r, g, b))  # Stack r, g, b across third dimension - create to 3D array (of R, G, B elements).

    # Convert from sRGB color space to LAB color space
    lab = rgb2lab(rgb)

    peaked_rgb = np.asarray([int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)], np.uint8)
    peaked_rgb = np.dstack((peaked_rgb[0], peaked_rgb[1], peaked_rgb[2]))
    peaked_lab = rgb2lab(peaked_rgb)

    # Compute Euclidean distance from peaked_lab to each element of lab
    lab_dist = np.sqrt(
        (lab[:, :, 0] - peaked_lab[:, :, 0])**2 + (lab[:, :, 1] - peaked_lab[:, :, 1])**2 + (lab[:, :, 2] - peaked_lab[:, :, 2])**2
    )

    # Get the index of the minimum distance
    min_index = lab_dist.argmin()

    # Get hex string of the color with the minimum Euclidean distance (minimum distance in LAB color space)
    peaked_closest_hex = hex_rgb_colors[min_index]

    # Get color name from the dictionary
    peaked_color_name = color_table[peaked_closest_hex]

    return peaked_color_name


def color_detection(img, n_colors, show_chart=False, output_chart=None):  # Gets a PNG image with alpha layer.
    img = img[img[:, :, 3] == 255][:, :3]

    clf = KMeans(n_clusters=n_colors)

    colors = clf.fit_predict(img)

    counts = Counter(colors)
    print('Counts: ', counts)
    center_colors = clf.cluster_centers_
    print('Center Colors: ', center_colors)

    ordered_colors = [center_colors[i] for i in range(n_colors)]
    hex_colors = [rgb2hex(ordered_colors[i]) for i in range(n_colors)]
    rgb_colors = [ordered_colors[i] for i in range(n_colors)]

    color_category = dict()

    for idx, hex_color in enumerate(hex_colors):
        color_category[hex_color] = counts[idx]

    print('Color Category: ', color_category)
    print('Color Hex Codes: ', hex_colors)
    print('Color RGB: ', rgb_colors)

    if (show_chart):
        plt.figure(figsize=[10, 10])
        plt.pie(color_category.values(), labels=color_category.keys(), colors=color_category.keys())
        plt.savefig(output_chart)

    return color_category

from PIL import Image

def color_detection2(img, n_colors, show_chart=False, output_chart=None):
    # Convert RGBA image to RGB
    img = Image.fromarray(img.astype('uint8'), 'RGBA')
    img = img.convert('RGB')
    clf = KMeans(n_clusters=n_colors)

    colors = clf.fit_predict(img)

    counts = Counter(colors)
    print('Counts: ', counts)
    center_colors = clf.cluster_centers_
    print('Center Colors: ', center_colors)

    ordered_colors = [center_colors[i] for i in range(n_colors)]
    hex_colors = [rgb2hex(ordered_colors[i]) for i in range(n_colors)]
    rgb_colors = [ordered_colors[i] for i in range(n_colors)]

    color_category = dict()

    for idx, hex_color in enumerate(hex_colors):
        color_category[hex_color] = counts[idx]

    print('Color Category: ', color_category)
    print('Color Hex Codes: ', hex_colors)
    print('Color RGB: ', rgb_colors)

    if (show_chart):
        plt.figure(figsize=[10, 10])
        plt.pie(color_category.values(), labels=color_category.keys(), colors=color_category.keys())
        plt.savefig(output_chart)

    return color_category

# if __name__ == '__main__':
#     input_path = sys.argv[1]
#     output_chart = sys.argv[2]

#     image = Image.open(input_path)

#     output = color_detection(image, n_colors=3, show_chart=True, output_chart=output_chart)


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_chart = sys.argv[2]

    image = np.array(Image.open(input_path))

    output = color_detection(image, n_colors=3, show_chart=True, output_chart=output_chart)
    # print(hex2name('#8c7770'))
