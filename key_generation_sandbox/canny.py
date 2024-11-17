from math import sqrt, atan2, pi

import cv2
import numpy as np
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from collections import defaultdict
import utils
def canny_edge_detector(input_image, blur_power):
    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transform the image to grayscale
    grayscaled = (compute_grayscale(input_pixels, width, height, 0), compute_grayscale(input_pixels, width, height, 1), compute_grayscale(input_pixels, width, height, 2))


    # Blur it to remove noise
    blurred = (compute_blur(grayscaled[0], blur_power), compute_blur(grayscaled[1], blur_power), compute_blur(grayscaled[2], blur_power))

    # Compute the gradient
    gradient = compute_gradient(blurred[0], width, height) + compute_gradient(blurred[1], width, height) + compute_gradient(blurred[2], width, height)

    gradient = normalize_gradient(gradient, 10)

    # Non-maximum suppression
    #filter_out_non_maximum(gradient, direction, width, height)

    # Filter out some edges
    keep = filter_strong_edges(gradient, width, height, 15, 20)

    return keep


def compute_grayscale(input_pixels, width, height, num):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[num]) // 3
    return grayscale


def compute_blur(input_pixels, blur_power):
    return cv2.GaussianBlur(input_pixels, (blur_power, blur_power) ,0)


def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                direction[x, y] = atan2(magy, magx)
    #return gradient, direction
    return gradient

def normalize_gradient(gradient, brightness):
    req_brightness = brightness * gradient.shape[0] * gradient.shape[1]
    brightness = 0
    for x in range(gradient.shape[0]):
        for y in range(gradient.shape[1]):
            brightness = brightness + gradient[x, y]
    br_coef = req_brightness / brightness

    new_gradient = gradient
    for x in range(gradient.shape[0]):
        for y in range(gradient.shape[1]):
            new_gradient[x, y] = int(gradient[x, y] * br_coef)
    return new_gradient

def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0


def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)

def find_circles(input_image, rmin, rmax, precision, blur_power):
    steps = 30
    threshold = precision

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x, y in canny_edge_detector(input_image, blur_power):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold:
            #print(v / steps, x, y, r)
            circles.append((x, y, r, v/steps))
    return circles

def rotate(input_image, rotation, r):
    rotated_image = Image.new("RGB", [2 * r + 1, 2 * r + 1])
    input_pixels = input_image.load()
    draw_result = ImageDraw.Draw(rotated_image)
    for x1 in range(-r, r + 1):
        for y1 in range(-r, r + 1):
            if (x1) ** 2 + (y1) ** 2 <= r ** 2:
                draw_result.point((x1 + r, y1 + r),
                                  input_pixels[int(r + (x1 * cos(rotation) + y1 * sin(rotation))),
                                  int(r + (-x1 * sin(rotation) + y1 * cos(rotation)))])
            else:
                draw_result.point((x1 + r, y1 + r), (0, 0, 0))
    return rotated_image