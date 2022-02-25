import matplotlib.pyplot as plt
import colorsys
import random
import matplotlib.patches as patches
import numpy as np

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, boxes, figsize=(12, 12), ax=None):
    #input:
    #image is PIL; boxes is numpy array

    if isinstance(image, np.ndarray) == False:
        image = np.array(image)
    if isinstance(boxes, np.ndarray) == False:   
        boxes = boxes.numpy()

    # Number of instances
    N = len(boxes)
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    #ax.set_title(title)
    for i in range(N):
        color = colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
    ax.imshow(image, cmap="gray")
    plt.show()
    return ax
