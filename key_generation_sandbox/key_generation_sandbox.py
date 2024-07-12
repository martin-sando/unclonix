#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imagehash
from math import sqrt, pi, cos, sin, exp
from canny import return_grayscale, find_circles, rotate, compute_gradient
from collections import defaultdict
import sys
import os.path
import numpy as np
from findiff import Gradient, Divergence, Laplacian, Curl
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray


input_folder = '../input'
output_folder = '../output'
bloblist_folder = output_folder + '/bloblist'

def check_inside(x, y, w, h, overflow=0, rd=0):
    return (((x - rd) >= -overflow) & ((y - rd) >= -overflow)) & (
            ((x + rd) < (w + overflow)) & ((y + rd) < (h + overflow)))


def get_best_circle(circles, width, height, max_overflow):
    x0, y0, r0, pr0 = 0, 0, 0, 0
    for circle in circles:
        x = circle[0]
        y = circle[1]
        r = circle[2]
        pr = circle[3]
        if ((pr > pr0) | ((pr == pr0) & (r > r0))) & check_inside(x, y, width, height, max_overflow, r):
            x0, y0, r0, pr0 = circle

    if r0 == 0:
        return (0, 0, 0, 0)
    return (x0, y0, r0, pr0)


def get_circle_image(x0, y0, r0, compression_power, saved_image):
    x0, y0 = x0 * compression_power + compression_power // 2, y0 * compression_power + compression_power // 2
    saved_pixels = saved_image.load()
    circle_image = Image.new("RGB", [2 * r0 + 1, 2 * r0 + 1])
    draw_result = ImageDraw.Draw(circle_image)
    for x1 in range(-r0, r0 + 1):
        for y1 in range(-r0, r0 + 1):
            if (check_inside(x1 + x0, y1 + y0, saved_image.width, saved_image.height)):
                draw_result.point((x1 + r0, y1 + r0), saved_pixels[x1 + x0, y1 + y0])
            else:
                draw_result.point((x1 + r0, y1 + r0), (0, 0, 0))
    return circle_image


def get_fft_image(image, image_size, fft_radius, coef_1, coef_2):
    array_image = to_array(image)

    fft_image = Image.new("RGB", [2 * fft_radius + 1, 2 * fft_radius + 1])
    draw_result = ImageDraw.Draw(fft_image)
    fft_transform = np.fft.fft2(array_image, None, None, "backward")
    fft_transform = np.fft.fftshift(fft_transform)

    for x1 in range(-fft_radius, fft_radius + 1):
        for y1 in range(-fft_radius, fft_radius + 1):
            a = fft_transform[x1 + image_size // 2, y1 + image_size // 2].real
            b = fft_transform[x1 + image_size // 2, y1 + image_size // 2].imag
            color1 = int(np.log(sqrt(a ** 2 + b ** 2)) * coef_1 + coef_2)
            draw_result.point((x1 + fft_radius, y1 + fft_radius), (color1, color1, color1))
    return fft_image

def get_gradient_image(image, width, height):
    input_pixels = image.load()
    grad_image = Image.new("RGB", [width, height])
    draw_result = ImageDraw.Draw(grad_image)

    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y][0] - input_pixels[x - 1, y][0]
                magy = input_pixels[x, y + 1][0] - input_pixels[x, y - 1][0]
                color = int(sqrt(magx**2 + magy**2))
                draw_result.point((x, y), (color, color, color))
    return grad_image

def find_small_circles(input_image, rmin, rmax, precision):
    steps = 10
    threshold = precision * 255
    input_pixels = input_image.load()

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x in range(input_image.width):
        for y in range(input_image.height):
            brightness = input_pixels[x, y][0]
            for r, dx, dy in points:
                a = x - dx
                b = y - dy
                acc[(a, b, r)] += brightness

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > (rc * 1.5) ** 2 for xc, yc, rc, pr in circles):
            #print(v / steps, x, y, r)
            circles.append((x, y, r, v/steps))
    return circles

def to_array(input_image):
    array_image = np.ndarray((input_image.width, input_image.height))
    image_pixels = input_image.load()

    for x1 in range(input_image.width):
        for y1 in range(input_image.height):
            array_image[x1, y1] = image_pixels[x1, y1][0]
    return array_image

def to_image(input_array, width, length):
    image = Image.new("RGB", [width, length])
    draw_result = ImageDraw.Draw(image)
    for x1 in range(width):
        for y1 in range(length):
            color = int(input_array[x1, y1])
            draw_result.point((x1, y1), (color, color, color))
    return image


def process_file(input_file):
    filename = input_file.split('.')[0]
    print('Processing ' + filename)
    white = (255, 255, 255)
    black = (0, 0, 0)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    req_width = 1024
    req_height = 1024
    req_size = (req_width, req_height)

    input_image = Image.open(os.path.join(input_folder, input_file))
    saved_image = input_image.copy()

    log_picture_number = 0

    def save(image, tag):
        nonlocal log_picture_number
        log_picture_number += 1
        tag = 'p' + str(log_picture_number).zfill(2) + tag
        image.save(os.path.join(output_folder, filename + "_" + tag + ".png"))
        image.save(os.path.join(output_folder, tag + "_" + filename + ".png"))


    save(input_image, 'input')
    width, height = input_image.width, input_image.height
    compression_power = width // 100
    #to_gray
    input_image = return_grayscale(input_image, width, height)

    save(input_image, 'todo1')

    #compress image copy to effectively find a circle
    input_image = input_image.resize((width // compression_power, height // compression_power))
    save(input_image, 'todo2')

    width, height = input_image.width, input_image.height

    input_pixels = input_image.load()

    # Find circles, assuming their D is at least quarter of min(h, w)
    rmin = int(min(input_image.height, input_image.width) / 8)
    rmax = int(min(input_image.height, input_image.width) / 1.9)
    precision = 0.7
    circles = find_circles(input_image, rmin, rmax, precision)

    if not circles:
        print('Error: not found circles')
        return

    max_overflow = 15
    x0, y0, r0, pr0 = get_best_circle(circles, width, height, max_overflow)

    if r0 == 0:
        print('Error: r0 = 0')
        return

    #extrapolating found circle to original scale
    logo_image = get_circle_image(x0, y0, int(r0 * compression_power * 1.1), compression_power, saved_image)


    save(logo_image, 'todo3')
    width = logo_image.width
    height = logo_image.height
    compression_power = width // 200
    saved_logo = logo_image.copy()
    logo_image = logo_image.resize((width // compression_power, height // compression_power))
    width = logo_image.width
    height = logo_image.height
    rmin = int(min(width, height) / 2.7)
    rmax = int(min(width, height) / 1.9)
    precision = 0.7
    circles = find_circles(logo_image, rmin, rmax, precision)

    if not circles:
        print('Error: not found circles (extremely weird!)')
        return

    max_overflow = 15
    x0, y0, r0, pr0 = get_best_circle(circles, width, height, max_overflow)

    x0, y0, r0 = x0 * compression_power + compression_power // 2, y0 * compression_power + compression_power // 2, int(r0 * compression_power - compression_power)
    width, height = saved_logo.width, saved_logo.height
    if r0 == 0:
        print('Error: r0 = 0')
        return

    saved_pixels = saved_logo.load()

    inner_colours = [[], [], []]

    for x1 in range(-r0, r0 + 1):
        for y1 in range(-r0, r0 + 1):
            if ((x1) ** 2 + (y1) ** 2 <= r0 ** 2) & check_inside(r0 + x1, r0 + y1, r0, r0):
                inner_colours[0].append(saved_pixels[x0 + x1, y0 + y1][0])
                inner_colours[1].append(saved_pixels[x0 + x1, y0 + y1][1])
                inner_colours[2].append(saved_pixels[x0 + x1, y0 + y1][2])

    inner_colours[0].sort()
    inner_colours[1].sort()
    inner_colours[2].sort()

    median = (inner_colours[0][len(inner_colours[0]) // 2],
              inner_colours[1][len(inner_colours[1]) // 2],
              inner_colours[2][len(inner_colours[2]) // 2])
    circled_image = Image.new("RGB", [2 * r0 + 1, 2 * r0 + 1])
    draw_result = ImageDraw.Draw(circled_image)

    #redrawing image to make circle big and erase blackground
    for x1 in range(-r0, r0 + 1):
        for y1 in range(-r0, r0 + 1):
            dist = sqrt((x1 ** 2 + y1 ** 2) / (r0 ** 2))
            if check_inside(x1 + r0, y1 + r0, width, height):
                color = (abs(median[0] - saved_pixels[x1 + x0, y1 + y0][0]) +
                         abs(median[1] - saved_pixels[x1 + x0, y1 + y0][1]) +
                         abs(median[2] - saved_pixels[x1 + x0, y1 + y0][2])) // 3
                if dist >= 1:
                    draw_result.point((x1 + r0, y1 + r0), black)
                elif dist < 0.9:
                    draw_result.point((x1 + r0, y1 + r0), (color, color, color))
                else:
                    color = int(color * (10.0 - 10.0 * dist))
                    draw_result.point((x1 + r0, y1 + r0), (color, color, color))

    save(circled_image, 'todo4')

    #resize to 1024*1024
    circled_image = circled_image.resize(req_size)
    circled_image = circled_image.copy()
    save(circled_image, 'last')
    circled_pixels = circled_image.load()


    morph_image = rgb2gray(circled_image)
    blobs_log = blob_log(morph_image, min_sigma = req_width / 400, max_sigma = req_width / 170, num_sigma=10, threshold=.03)
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

    blobs_dog = blob_dog(morph_image, min_sigma = 1.2, max_sigma = req_width / 170, threshold=.03)
    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
    blobs_doh = blob_doh(morph_image, max_sigma=20, threshold=.01)

    log_picture = circled_image.copy()
    draw_result = ImageDraw.Draw(log_picture)

    text_file = open(os.path.join(bloblist_folder, filename + '.txt'), 'w')
    for blob in blobs_log:
        x = blob[1]
        y = blob[0]
        sigma = blob[2]
        text_file.write(str(x) + ' ' + str(y) + ' ' + str(sigma) + "\n")
        draw_result.point((x, y), blue)
        for i in range(int(sigma)):
            draw_result.point((x, y+i), blue)
            draw_result.point((x, y-i), blue)
            draw_result.point((x+i, y), blue)
            draw_result.point((x-i, y), blue)
    save(log_picture, 'found_blobs_log')


    neighbor_array = np.ndarray((req_width, req_height))
    radius = 50
    for blob in blobs_log:
        x = blob[1]
        y = blob[0]
        sigma = blob[2]
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if check_inside(x+i, y+j, req_height, req_width):
                    dist = sqrt((i)**2 + (j) ** 2)
                    if dist <= radius ** 2:
                        neighbor_array[int(x+i), int(y+j)] += exp(-(dist + sigma)*0.1)*40

    neighbor_image = to_image(neighbor_array, req_width, req_height)
    save(neighbor_image, 'neighbors')




    dog_picture = circled_image.copy()
    draw_result = ImageDraw.Draw(dog_picture)
    for blob in blobs_dog:
        x = blob[1]
        y = blob[0]
        sigma = blob[2]
        draw_result.point((x, y), blue)
        for i in range(int(sigma)):
            draw_result.point((x, y+i), red)
            draw_result.point((x, y-i), red)
            draw_result.point((x+i, y), red)
            draw_result.point((x-i, y), red)
    save(dog_picture, 'found_blobs_dog')


    x, y = [np.linspace(0, req_width - 1, req_width)] * 2
    dx, dy = [c[1] - c[0] for c in (x, y)]
    lap = Laplacian(h=[dx, dy])

    circle_array = to_array(circled_image)
    lap_array = lap(circle_array) * 5
    lap_image = to_image(lap_array, req_width, req_height)


    save(lap_image, 'laplacian')

    # circled_array = np.empty((req_width, req_height))
    # for x in range(req_width):
    #     for y in range(req_height):
    #         circled_array[x, y] = circled_pixels[x, y][0]

    # fft_image = get_fft_image(circled_image, req_height, 15, 140, -1400)
    # save(fft_image, 'fft')
    #
    # grad_image = get_gradient_image(circled_image, req_width, req_height)
    # save(grad_image, 'gradient'))


    # draw_result = ImageDraw.Draw(circled_image)
    # found_glitter = find_small_circles(grad_image, 2, 5, 0.15)
    # for circle in found_glitter:
    #     x = circle[0]
    #     y = circle[1]
    #     draw_result.point((x, y), blue)
    #     for i in range(circle[2]):
    #         draw_result.point((x, y+i), blue)
    #         draw_result.point((x, y-i), blue)
    #         draw_result.point((x+i, y), blue)
    #         draw_result.point((x-i, y), blue)
    #
    #
    # save(circled_image, 'found_blobs')

    array_image = np.ndarray((req_width, req_height))
    for x1 in range(req_width):
        for y1 in range(req_height):
            array_image[x1, y1] = circled_pixels[x1, y1][0] + 0.0

    dct_array = cv2.dct(array_image)

    dct_size = 32
    dct_image = Image.new("RGB", [dct_size, dct_size])
    draw_result = ImageDraw.Draw(dct_image)
    for x1 in range(dct_size):
        for y1 in range(dct_size):
            color = int(dct_array[x1, y1])
            draw_result.point((x1, y1), (color, color, color))

    save(dct_image, 'dct')

    phash = imagehash.phash(circled_image)
    hash_as_str = str(phash)
    print(hash_as_str)
    # circled_image = rotate(circled_image, 100, 256)
    # save(circled_image, 'todo5')
    # phash = imagehash.phash(circled_image)
    # hash_as_str = str(phash)
    # print(hash_as_str)


def run_all():
    input_files = os.listdir(input_folder)
    for input_file in input_files:
        process_file(input_file)


if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bloblist_folder, exist_ok=True)
    run_all()
