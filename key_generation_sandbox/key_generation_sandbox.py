#!/usr/bin/env python3
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imagehash
from math import sqrt, pi, cos, sin, exp, atan2
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
                color = int(sqrt(magx ** 2 + magy ** 2))
                draw_result.point((x, y), (color, color, color))
    return grad_image


def get_variance_image(rad_min, rad_max, image, req_width, req_height):
    def get_queries(rad_min, rad_max):
        queries = []
        query = [0, 0, 0]
        for i in range(-rad_max - 1, rad_max + 2):
            for j in range(-rad_max - 1, rad_max + 2):
                dist = (i ** 2 + j ** 2) / (rad_max ** 2)
                dist2 = 2
                if rad_min > 0:
                    dist2 = (i ** 2 + j ** 2) / (rad_min ** 2)
                if (dist <= 1 <= dist2):
                    if (query == [0, 0, 0]):
                        query = [i, j, 0]
                else:
                    if query != [0, 0, 0]:
                        query[2] = j
                        queries.append(query)
                        query = [0, 0, 0]
        return queries

    def get_dots(rmin, rmax):
        ans = 0
        for i in range(-rmax - 1, rmax + 2):
            for j in range(-rmax - 1, rmax + 2):
                dist = (i ** 2 + j ** 2) / (rmax ** 2)
                dist2 = 2
                if rmin > 0:
                    dist2 = (i ** 2 + j ** 2) / (rmin ** 2)
                if (dist <= 1 <= dist2):
                    ans = ans + 1
        return ans

    array_img = to_array(image)
    array_img = array_img.copy()
    sq_array = array_img.copy()
    for i in range(array_img.shape[0]):
        for j in range(array_img.shape[1]):
            sq_array[i, j] = array_img[i, j] ** 2

    prefix_array = array_img.copy()
    for i in range(array_img.shape[0]):
        for j in range(array_img.shape[1]):
            if j == 0:
                prefix_array[i, j] = array_img[i, j]
            else:
                prefix_array[i, j] = prefix_array[i, j - 1] + array_img[i, j]

    sq_prefix_array = sq_array.copy()
    for i in range(array_img.shape[0]):
        for j in range(array_img.shape[1]):
            if j == 0:
                sq_prefix_array[i, j] = sq_array[i, j]
            else:
                sq_prefix_array[i, j] = sq_prefix_array[i, j - 1] + sq_array[i, j]

    rad_min = 32
    rad_max = 40
    query_set = get_queries(rad_min, rad_max)
    total_dots = get_dots(rad_min, rad_max)
    variance_image = Image.new("RGB", (req_width, req_height))
    draw_variance = ImageDraw.Draw(variance_image)
    for x1 in range(req_width):
        for y1 in range(req_height):
            sum = 0
            sq_sum = 0
            current_dots = total_dots
            for a, b1, b2 in query_set:
                if x1 + a < 0:
                    current_dots = current_dots + x1 + a
                    a = -x1
                if x1 + a >= req_width:
                    current_dots = current_dots + req_width - x1 - a - 1
                    a = req_width - x1 - 1
                if y1 + b1 < 0:
                    current_dots = current_dots + y1 + b1
                    b1 = -y1
                if y1 + b1 >= req_height:
                    current_dots = current_dots + req_height - y1 - b1
                    b1 = req_height - y1 - 1
                if y1 + b2 < 0:
                    current_dots = current_dots + y1 + b2
                    b2 = -y1
                if y1 + b2 >= req_height:
                    current_dots = current_dots + req_height - y1 - b2
                    b2 = req_height - y1 - 1
                sum = sum + prefix_array[x1 + a, y1 + b2] - prefix_array[x1 + a, y1 + b1]
                sq_sum = sq_sum + sq_prefix_array[x1 + a, y1 + b2] - sq_prefix_array[x1 + a, y1 + b1]
            sum = sum / total_dots
            sq_sum = sq_sum / total_dots
            variance = sq_sum - sum * sum
            color = int(variance) // 10
            draw_variance.point((x1, y1), (color, color, color))
    return variance_image


def get_partial_image(image, r, req_width, req_height):
    blue = (0, 0, 255)
    pixels = image.load()
    part_size = 128
    parts_image = Image.new("RGB", ((req_width // part_size), (req_height // part_size)))
    draw_parts = ImageDraw.Draw(parts_image)
    for i in range(req_width // part_size):
        for j in range(req_height // part_size):
            up = i * part_size
            down = i * part_size + part_size
            left = j * part_size
            right = j * part_size + part_size

            dist1 = sqrt((up - r) ** 2 + (left - r) ** 2) / r
            dist2 = sqrt((down - r) ** 2 + (left - r) ** 2) / r
            dist3 = sqrt((up - r) ** 2 + (right - r) ** 2) / r
            dist4 = sqrt((down - r) ** 2 + (right - r) ** 2) / r
            if (dist1 <= 1 and dist2 <= 1) and (dist3 <= 1 and dist4 <= 1):
                discr = 0
                for i1 in range(up, down):
                    for j1 in range(left, right):
                        d = i1 + j1 - up - left - (part_size - 1)
                        discr += d * pixels[i1, j1][0]
                color = (discr // 50000) + 128
                draw_parts.point((i, j), (color, color, color))
            else:
                draw_parts.point((i, j), blue)
    return parts_image


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
        if v / steps >= threshold and all(
                (x - xc) ** 2 + (y - yc) ** 2 > (rc * 1.5) ** 2 for xc, yc, rc, pr in circles):
            #print(v / steps, x, y, r)
            circles.append((x, y, r, v / steps))
    return circles


def to_array(input_image):
    array_image = np.zeros((input_image.width, input_image.height))
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


def get_field_image(input_image, width, height, precision, hru, low_b, up_b, cell, cutter_size=128, blobs=[]):
    text_file = open(os.path.join(bloblist_folder, 'dct' + '.txt'), 'w')
    r = width // 2
    result_array = np.zeros((width, height, 3))
    scores = np.zeros((11))
    dct_sum = np.zeros((128, 128))
    dct_sum[0, 0] = 1
    total_sum1 = 0
    total_sum2 = 0
    total_sum3 = 0
    for blob in blobs:
        coord_0 = int(blob[1])
        coord_1 = int(blob[0])
        dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
        if dist > 0.8:
            continue
        angle = atan2((coord_1 - r), (coord_0 - r))
        blob_img = input_image.crop(
            (coord_0 - cutter_size, coord_1 - cutter_size, coord_0 + cutter_size, coord_1 + cutter_size))
        rot_image = blob_img.rotate((angle * (180 / pi)))
        blob_img = rot_image.crop(
            (int(cutter_size / 2), int(cutter_size / 2), int(cutter_size * (3 / 2)), int(cutter_size * (3 / 2))))
        blob_pixels = blob_img.load()

        array_image = np.zeros((cutter_size, cutter_size))
        for x1 in range(cutter_size):
            for y1 in range(cutter_size):
                array_image[x1, y1] = blob_pixels[x1, y1][0] + 0.0

        dct_array = cv2.dct(array_image)
        dct_array[0][0] = 0
        for i in range(low_b, up_b + 1):
            for j in range(low_b, up_b + 1):
                dct_sum[i, j] = dct_sum[i, j] + abs(dct_array[i, j])

    for blob in blobs:
        coord_0 = int(blob[1])
        coord_1 = int(blob[0])
        dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
        if dist > 0.8:
            continue
        angle = atan2((coord_1 - r), (coord_0 - r))
        blob_img = input_image.crop(
            (coord_0 - cutter_size, coord_1 - cutter_size, coord_0 + cutter_size, coord_1 + cutter_size))
        rot_image = blob_img.rotate((angle * (180 / pi)))
        blob_img = rot_image.crop(
            (int(cutter_size / 2), int(cutter_size / 2), int(cutter_size * (3 / 2)), int(cutter_size * (3 / 2))))
        blob_pixels = blob_img.load()

        array_image = np.zeros((cutter_size, cutter_size))
        for x1 in range(cutter_size):
            for y1 in range(cutter_size):
                array_image[x1, y1] = blob_pixels[x1, y1][0] + 0.0

        dct_array = cv2.dct(array_image)
        dct_array[0][0] = 0
        score = [0, 0, 0]
        for i in range(low_b, up_b + 1):
            for j in range(low_b, up_b + 1):
                score[0] += hru[0][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (dct_array[i, j] / dct_sum[i, j])
                total_sum1 += hru[0][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * abs(dct_array[i, j] / dct_sum[i, j])
                score[1] += hru[1][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (dct_array[i, j] / dct_sum[i, j])
                total_sum2 += hru[1][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * abs(dct_array[i, j] / dct_sum[i, j])
                score[2] += hru[2][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (dct_array[i, j] / dct_sum[i, j])
                total_sum3 += hru[2][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * abs(dct_array[i, j] / dct_sum[i, j])

        result_array[coord_0, coord_1, 0] = score[0]
        result_array[coord_0, coord_1, 1] = score[1]
        result_array[coord_0, coord_1, 2] = score[2]
    field_image = input_image.copy()
    draw_result = ImageDraw.Draw(field_image)
    for cnt in scores:
        print(cnt, end=' ')

    scaling1 = total_sum1 / (len(blobs) * 128)
    scaling2 = total_sum2 / (len(blobs) * 128)
    scaling3 = total_sum3 / (len(blobs) * 128)
    for blob in blobs:
        x = int(blob[1])
        y = int(blob[0])
        colors = (int(result_array[x, y][0] / scaling1), int(result_array[x, y][1] / scaling2), int(result_array[x, y][2] / scaling3))
        draw_result.point((x, y), colors)
        for i in range(int(blob[2])):
            draw_result.point((x, y + i), colors)
            draw_result.point((x, y - i), colors)
            draw_result.point((x + i, y), colors)
            draw_result.point((x - i, y), colors)

    return field_image


def get_angle_image(input_image, width, height, precision, mode, cutter_size=64):
    r = width // 2
    result_array = np.zeros((width, height))
    for x in range(width // precision):
        for y in range(height // precision):
            coord_0 = x * precision
            coord_1 = y * precision
            dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
            if dist > 0.7:
                continue
            angle = atan2((coord_1 - r), (coord_0 - r))
            blob_img = input_image.crop(
                (coord_0 - cutter_size, coord_1 - cutter_size, coord_0 + cutter_size, coord_1 + cutter_size))
            rot_image = blob_img.rotate((angle * (180 / pi)))
            blob_img = rot_image.crop(
                (int(cutter_size / 2), int(cutter_size / 2), int(cutter_size * (3 / 2)), int(cutter_size * (3 / 2))))
            blob_pixels = blob_img.load()

            array_image = np.zeros((cutter_size, cutter_size))
            for x1 in range(cutter_size):
                for y1 in range(cutter_size):
                    array_image[x1, y1] = blob_pixels[x1, y1][0] + 0.0

            score1 = 0
            if mode == "r" or mode == "both":
                for x1 in range(cutter_size):
                    for y1 in range(cutter_size // 2):
                        y2 = cutter_size - 1 - y1
                        score1 += abs(array_image[x1, y1] - 128) + abs(array_image[x1, y2] - 128) - 2.5 * abs(
                            array_image[x1, y1] - array_image[x1, y2])
                score1 = max(score1, 0)

            score2 = 0
            if mode == "arc" or mode == "both":
                for x1 in range(cutter_size // 2):
                    for y1 in range(cutter_size):
                        x2 = cutter_size - 1 - x1
                        score2 += abs(array_image[x1, y1] - 128) + abs(array_image[x2, y1] - 128) - 2.5 * abs(
                            array_image[x1, y1] - array_image[x2, y1])
                score2 = max(score2, 0)
            result_array[coord_0, coord_1] = min(score1, score2)


    for x in range(width // precision):
        for y in range(height // precision):
            for x1 in range(precision):
                for y1 in range(precision):
                    if x1 + y1 == 0:
                        continue
                    coord_0 = x * precision + x1
                    coord_1 = y * precision + y1
                    dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
                    if dist > 0.7:
                        continue
                    comp_1 = result_array[x * precision, y * precision] * (1 - x1 / precision) * (1 - y1 / precision)
                    comp_2 = result_array[x * precision, y * precision + precision] * (1 - x1 / precision) * (
                            y1 / precision)
                    comp_3 = result_array[x * precision + precision, y * precision] * (x1 / precision) * (
                            1 - y1 / precision)
                    comp_4 = result_array[x * precision + precision, y * precision + precision] * (x1 / precision) * (
                            y1 / precision)
                    result_array[coord_0, coord_1] = comp_1 + comp_2 + comp_3 + comp_4
    field_image = Image.new("RGB", [width, height])
    draw_result = ImageDraw.Draw(field_image)
    scaling = (cutter_size * cutter_size) * 0.1
    for x in range(width):
        for y in range(height):
            dist = sqrt((x - r) ** 2 + (y - r) ** 2) / r
            color = 0
            if dist < 0.7:
                color = int(result_array[x, y] / scaling)
            draw_result.point((x, y), (color, color, color))
    return field_image


def process_file(input_file, full_research_mode):
    random.seed(566)
    hru_array = []
    for i in range(10):
        hru = [random.gauss(mu=0.0, sigma=1.0) for _ in range(64)]
        hru_array.append(hru)
    filename = input_file.split('.')[0]
    print('Processing ' + filename)
    white = (255, 255, 255)
    gray = (127, 127, 127)
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
    compression_power = width // 150
    #to_gray
    input_image = return_grayscale(input_image, width, height)

    save(input_image, 'to_gray')

    #compress image copy to effectively find a circle
    input_image = input_image.resize((width // compression_power, height // compression_power))
    save(input_image, 'compressed')

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

    save(logo_image, 'large_circle')
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

    x0, y0, r0 = x0 * compression_power + compression_power // 2, y0 * compression_power + compression_power // 2, int(
        r0 * compression_power * 0.98 - compression_power)
    width, height = saved_logo.width, saved_logo.height
    if r0 == 0:
        print('Error: r0 = 0')
        return

    saved_pixels = saved_logo.load()

    circled_image = Image.new("RGB", [2 * r0 + 1, 2 * r0 + 1])
    draw_result = ImageDraw.Draw(circled_image)

    #redrawing image to make circle big and erase blackground
    for x1 in range(-r0, r0 + 1):
        for y1 in range(-r0, r0 + 1):
            dist = sqrt((x1 ** 2 + y1 ** 2) / (r0 ** 2))
            if check_inside(x1 + x0, y1 + y0, width, height):
                if dist >= 1:
                    draw_result.point((x1 + r0, y1 + r0), black)
                else:
                    draw_result.point((x1 + r0, y1 + r0), saved_pixels[x1 + x0, y1 + y0])

    #resize

    circled_image = circled_image.resize(req_size)
    circled_image = circled_image.copy()
    saved_circle_image = circled_image.copy()
    save(circled_image, 'circled')
    saved_pixels = circled_image.load()
    draw_result = ImageDraw.Draw(circled_image)
    #redrawing image to make circle big and erase blackground
    r = (req_width / 2)

    new_circled_image = Image.new("RGB", req_size)
    draw_result = ImageDraw.Draw(new_circled_image)

    skip_factor = req_width // 150
    for x1 in range(req_width):
        for y1 in range(req_height):
            dist = sqrt((x1 - r) ** 2 + (y1 - r) ** 2) / r
            if dist >= 1:
                draw_result.point((x1, y1), black)
            else:
                surrounding_colours = [[], [], []]
                close_r = int((req_width * 0.02) / skip_factor)
                for dx0 in range(-close_r, close_r + 1):
                    for dy0 in range(-close_r, close_r + 1):
                        dx = dx0 * skip_factor
                        dy = dy0 * skip_factor
                        dist2 = ((x1 + dx - r) ** 2 + (y1 + dy - r) ** 2) / (r ** 2)
                        dist3 = (dx ** 2 + dy ** 2) / ((close_r * skip_factor) ** 2)
                        if ((dist2 <= 1) and (dist3 <= 1)) and check_inside(x1 + dx, y1 + dy, req_width, req_height):
                            surrounding_colours[0].append(saved_pixels[x1 + dx, y1 + dy][0])
                            surrounding_colours[1].append(saved_pixels[x1 + dx, y1 + dy][1])
                            surrounding_colours[2].append(saved_pixels[x1 + dx, y1 + dy][2])
                surrounding_colours[0].sort()
                surrounding_colours[1].sort()
                surrounding_colours[2].sort()
                median = (surrounding_colours[0][len(surrounding_colours[0]) // 2],
                          surrounding_colours[1][len(surrounding_colours[1]) // 2],
                          surrounding_colours[2][len(surrounding_colours[2]) // 2])
                color = (abs(median[0] - saved_pixels[x1, y1][0]) +
                         abs(median[1] - saved_pixels[x1, y1][1]) +
                         abs(median[2] - saved_pixels[x1, y1][2])) // 3
                if dist < 0.9:
                    draw_result.point((x1, y1), (color, color, color))
                else:
                    color = int(color * (10.0 - 10.0 * dist))
                    draw_result.point((x1, y1), (color, color, color))

    save(new_circled_image, 'trimmed')

    circled_pixels = new_circled_image.load()
    draw_result = ImageDraw.Draw(new_circled_image)

    total_brightness = 0
    for x1 in range(req_width):
        for y1 in range(req_height):
            total_brightness += circled_pixels[x1, y1][0]

    req_brightness = 15 * req_width * req_height
    brightness_coef = req_brightness / total_brightness
    for x1 in range(req_width):
        for y1 in range(req_height):
            color = int(circled_pixels[x1, y1][0] * brightness_coef)
            draw_result.point((x1, y1), (color, color, color))

    save(new_circled_image, 'brightened')

    new_circled_image = new_circled_image.copy()
    circled_pixels = new_circled_image.load()

    pairs = [(2, 4)]
    dots = []

    # for pair in pairs:
    #     field_image = get_field_image(new_circled_image, req_width, req_height, 10, hru, pair[0], pair[1], False)
    #     save(field_image, 'field_image_array' + str(pair[0]) + '..' + str(pair[1]))
    #     angle_image = get_angle_image(field_image, req_width, req_height, mode="both", precision=5)
    #     save(angle_image, 'angle_image_array' + str(pair[0]) + '..' + str(pair[1]))
    #
    # for dot in dots:
    #     field_image = get_field_image(new_circled_image, req_width, req_height, 10, hru, dot[0], dot[1], True)
    #     save(field_image, 'field_image_dot' + str(dot[0]) + '.' + str(dot[1]))
    #     angle_image = get_angle_image(field_image, req_width, req_height, mode="both", precision=5)
    #     save(angle_image, 'angle_image_array' + str(dot[0]) + '..' + str(dot[1]))

    morph_image = rgb2gray(new_circled_image)
    blobs_log = blob_log(morph_image, min_sigma=1, max_sigma=req_width / 190, num_sigma=10, threshold=.03,
                         overlap=0.5)
    blobs_log[:, 2] = (blobs_log[:, 2] * np.sqrt(2)) + 1

    field_image = get_field_image(new_circled_image, req_width, req_height, 10, hru_array, 0, 5, False, blobs = blobs_log)
    save(field_image, 'field_image_array' + '2' + '..' + '4')
    angle_image = get_angle_image(field_image, req_width, req_height, mode="both", precision=5)
    save(angle_image, 'angle_image_array' + '2' + '..' + '4')

    # filtered_image = Image.new("RGB", req_size)
    # draw_result = ImageDraw.Draw(filtered_image)
    # saved_pixels = angle_image.load()
    # for x1 in range(req_width):
    #     for y1 in range(req_height):
    #         if (saved_pixels[x1, y1][0] > 250):
    #             draw_result.point((x1, y1), white)
    #         elif (saved_pixels[x1, y1][0] > 200):
    #             for x2 in range(-10, 10):
    #                 for y2 in range(-10, 10):
    #                     if (saved_pixels[x1+x2, y1+y2][0] > 250):
    #                         draw_result.point((x1, y1), white)
    #
    # save(filtered_image, 'fitlered')



    morph_image = rgb2gray(angle_image)
    blobs_log = blob_log(morph_image, min_sigma=1, max_sigma=req_width / 190, num_sigma=10, threshold=.10,
                         overlap=0.5)
    blobs_log[:, 2] = (blobs_log[:, 2] * np.sqrt(2)) + 1

    dot_image = angle_image.copy()
    draw_result = ImageDraw.Draw(dot_image)
    transformed_blobs = []

    for blob in blobs_log:
        transformed_blobs.append((blob[1], blob[0], blob[2]))
    blobs_log = transformed_blobs
    for blob in blobs_log:
        x = blob[0]
        y = blob[1]
        sigma = blob[2]
        draw_result.point((x, y), blue)
        for i in range(int(sigma)):
            draw_result.point((x, y + i), blue)
            draw_result.point((x, y - i), blue)
            draw_result.point((x + i, y), blue)
            draw_result.point((x - i, y), blue)

    save(dot_image, "dots")

    best_score = 1000
    best_blobs = []
    for blob1 in blobs_log:
        for blob2 in blobs_log:
            if blob1[0] == blob2[0] and blob1[1] == blob2[1]:
                continue
            for blob3 in blobs_log:
                if (blob1[0] == blob3[0] and blob1[1] == blob3[1]) or (blob2[0] == blob3[0] and blob2[1] == blob3[1]):
                    continue
                coord1 = (blob1[0] - req_width // 2, blob1[1] - req_height // 2)
                coord2 = (blob2[0] - req_width // 2, blob2[1] - req_height // 2)
                coord3 = (blob3[0] - req_width // 2, blob3[1] - req_height // 2)
                dist1 = sqrt((coord1[0]) ** 2 + (coord1[1]) ** 2) / r
                dist2 = sqrt((coord2[0]) ** 2 + (coord2[1]) ** 2) / r
                dist3 = sqrt((coord3[0]) ** 2 + (coord3[1]) ** 2) / r
                dist_12 = sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
                if ((0.4 < dist1 < 0.6) and (0.4 < dist2 < 0.6)) and ((0.4 < dist3 < 0.6) and dist_12 > 300):
                    angle1 = atan2(coord2[1] - coord1[1], coord2[0] - coord1[0]) * (180 / pi)
                    angle2 = atan2(coord3[1] - coord2[1], coord3[0] - coord2[0]) * (180 / pi)
                    angle3 = atan2(coord1[1] - coord3[1], coord1[0] - coord3[0]) * (180 / pi)
                    angles = []
                    angler1 = 180 - angle1 + angle2
                    if angler1 > 360:
                        angler1 = angler1 - 360
                    if angler1 < 0:
                        angler1 = angler1 + 360
                    angler2 = 180 - angle1 + angle3
                    if angler2 > 360:
                        angler2 = angler2 - 360
                    if angler2 < 0:
                        angler2 = angler2 + 360
                    angler3 = 180 - angle3 + angle2
                    if angler3 > 360:
                        angler3 = angler3 - 360
                    if angler3 < 0:
                        angler3 = angler3 + 360
                    angles.append(min(angler1, 360 - angler1))
                    angles.append(min(angler2, 360 - angler2))
                    angles.append(min(angler3, 360 - angler3))
                    angles.sort()
                    cur_score = abs(angles[0] - 55) + abs(angles[1] - 60) + abs(angles[2] - 65)
                    if cur_score < best_score:
                        best_score = cur_score
                        best_blobs = [coord1, coord2, coord3]

    triangle_image = angle_image.copy()

    draw_result = ImageDraw.Draw(triangle_image)
    for blob in best_blobs:
        x = blob[0] + req_width // 2
        y = blob[1] + req_width // 2
        draw_result.point((x, y), blue)
        for i in range(3):
            draw_result.point((x, y + i), blue)
            draw_result.point((x, y - i), blue)
            draw_result.point((x + i, y), blue)
            draw_result.point((x - i, y), blue)

    save(triangle_image, "triangle")

    if not full_research_mode:
        return

    sector_size = 128
    sector_i = 4
    sector_j = 4
    sector_image = Image.new("RGB", [sector_size, sector_size])
    draw_sector = ImageDraw.Draw(sector_image)
    for i in range(sector_i * sector_size, sector_i * sector_size + sector_size):
        for j in range(sector_j * sector_size, sector_j * sector_size + sector_size):
            draw_sector.point((i - sector_i * sector_size, j - sector_j * sector_size), circled_pixels[i, j])
    save(sector_image, "sector")

    compress_size = 8
    sector_pixels = sector_image.load()
    save(sector_image, "sector_compressed")

    dct_size = 8

    array_sector = np.zeros((sector_size, sector_size))
    for x1 in range(sector_size):
        for y1 in range(sector_size):
            array_sector[x1, y1] = sector_pixels[x1, y1][0] + 0.0
    dct_array = cv2.dct(array_sector)
    dct_image = Image.new("RGB", [dct_size, dct_size])
    draw_result = ImageDraw.Draw(dct_image)
    for x1 in range(dct_size):
        for y1 in range(dct_size):
            color = int(dct_array[x1, y1])
            draw_result.point((x1, y1), (color, color, color))

    save(dct_image, 'dct_sector')

    morph_image = rgb2gray(new_circled_image)
    blobs_log = blob_log(morph_image, min_sigma=req_width / 450, max_sigma=req_width / 190, num_sigma=10, threshold=.03,
                         overlap=0.5)
    blobs_log[:, 2] = (blobs_log[:, 2] * np.sqrt(2)) + 1

    # blobs_dog = blob_dog(morph_image, min_sigma=1.2, max_sigma=req_width / 170, threshold=.03)
    # blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
    # blobs_doh = blob_doh(morph_image, max_sigma=20, threshold=.01)

    log_picture = new_circled_image.copy()
    draw_result = ImageDraw.Draw(log_picture)

    draw_result_bright = ImageDraw.Draw(saved_circle_image)

    text_file = open(os.path.join(bloblist_folder, filename + '.txt'), 'w')

    brightened_blobs = []
    for blob in blobs_log:
        x = blob[1]
        y = blob[0]
        sigma = blob[2]
        brightness = 0
        for i in range(-int(sigma), int(sigma) + 1):
            for j in range(-int(sigma), int(sigma) + 1):
                if check_inside(x + i, y + j, req_height, req_width):
                    dist = sqrt((i) ** 2 + (j) ** 2)
                    if dist <= sigma ** 2:
                        brightness += exp(-(dist / (2 * sigma))) * circled_pixels[(x, y)][0]
        brightness = brightness / (2 * sigma * sigma)
        brightened_blobs.append((x, y, sigma, brightness))

    for blob in brightened_blobs:
        x = blob[0]
        y = blob[1]
        sigma = blob[2]
        brightness = blob[3]
        text_file.write(str(x) + ' ' + str(y) + ' ' + str(sigma) + ' ' + str(brightness) + "\n")
        draw_result.point((x, y), blue)
        for i in range(int(sigma)):
            draw_result.point((x, y + i), blue)
            draw_result.point((x, y - i), blue)
            draw_result.point((x + i, y), blue)
            draw_result.point((x - i, y), blue)
            draw_result_bright.point((x, y + i), blue)
            draw_result_bright.point((x, y - i), blue)
            draw_result_bright.point((x + i, y), blue)
            draw_result_bright.point((x - i, y), blue)
    save(log_picture, 'blobs_log')
    save(saved_circle_image, 'blobs_log_colored')
    save(log_picture, 'last')

    counter = 0

    dct_arrays = []
    good_blobs = []
    pigs = []
    for blob in brightened_blobs:
        dist = sqrt((blob[0] - r) ** 2 + (blob[1] - r) ** 2) / r
        if dist < 0.2 or dist > 0.8:
            continue
        angle = atan2((blob[1] - r), (blob[0] - r))

        if blob[1] - r < 0:
            angle += pi

        blob_img = new_circled_image.crop((blob[0] - 128, blob[1] - 128, blob[0] + 128, blob[1] + 128))
        rot_image = blob_img.rotate(360 - angle * (180 / pi))
        blob_img = rot_image.crop((64, 64, 192, 192))
        blob_pixels = blob_img.load()

        array_image = np.zeros((128, 128))
        for x1 in range(128):
            for y1 in range(128):
                array_image[x1, y1] = blob_pixels[x1, y1][0] + 0.0

        dct_array = cv2.dct(array_image)

        pig = 0
        dct_size = 8
        for i in range(3):
            for j in range(3):
                pig = pig + dct_array[i + 4, j + 4] * hru[i * 3 + j]

        dct_arrays.append(pig)
        pigs.append(pig)
        good_blobs.append((blob[0], blob[1], blob[2], blob[3], pig))

    log_picture = new_circled_image.copy()
    draw_result = ImageDraw.Draw(log_picture)
    pigs.sort()
    denominator = pigs[len(pigs) - 5]

    def linear_score(this, worst, best):
        if this < worst:
            return 0
        if this > best:
            return 1
        return (this - worst) / (best - worst)

    def linear_score2(this, worst1, best1, best2, worst2):
        if this < worst1:
            return 0
        if worst1 < this < best1:
            return (this - worst1) / (best1 - worst1)
        if best1 < this < best2:
            return 1
        if best2 < this < worst2:
            return (worst2 - this) / (worst2 - best2)
        return 0

    scored_blobs = []
    for blob in good_blobs:
        sigma = blob[2]
        dist = sqrt((blob[0] - r) ** 2 + (blob[1] - r) ** 2) / r
        score = linear_score(sigma, 5.5, 8)
        score = score + linear_score(blob[4], 300, 1200)
        score = score + linear_score2(dist, 0.45, 0.5, 0.7, 0.75)
        scored_blobs.append((blob[0], blob[1], blob[2], blob[3], blob[4], score))

    while True:
        new_blobs = []
        for blob in scored_blobs:
            cur_blob = blob
            for blob2 in scored_blobs:
                distance = (cur_blob[0] - blob2[0]) ** 2 + (cur_blob[1] - blob2[1]) ** 2
                if distance < 3000:
                    if cur_blob[5] < (blob2[5] - 0.1):
                        cur_blob = (0, 0, 0, 0, 0, 0)
                        break
            if cur_blob[0] != 0:
                new_blobs.append(cur_blob)
        if len(new_blobs) != len(scored_blobs):
            scored_blobs = new_blobs
        else:
            break

    for blob in scored_blobs:
        x = blob[0]
        y = blob[1]
        sigma = blob[2]
        brightness = blob[3]
        dist = sqrt((blob[0] - r) ** 2 + (blob[1] - r) ** 2) / r
        score = linear_score(sigma, 5.5, 8)
        score = score + linear_score(blob[4], 300, 1200)
        score = score + linear_score2(dist, 0.45, 0.5, 0.7, 0.75)
        color = (int(128 + blob[4] // 10), 0, int(128 - blob[4] // 10))
        text_file.write(
            str(x) + ' ' + str(y) + ' ' + str(sigma) + ' ' + str(brightness) + ' ' + str(blob[4]) + "\n")
        draw_result.point((x, y), color)
        for i in range(int(sigma)):
            draw_result.point((x, y + i), color)
            draw_result.point((x, y - i), color)
            draw_result.point((x + i, y), color)
            draw_result.point((x - i, y), color)
            draw_result_bright.point((x, y + i), color)
            draw_result_bright.point((x, y - i), color)
            draw_result_bright.point((x + i, y), color)
            draw_result_bright.point((x - i, y), color)
    save(log_picture, 'good_blobs_log')
    save(saved_circle_image, 'good_blobs_log_colored')

    blob_array = np.zeros((req_width, req_height))
    for blob in brightened_blobs:
        x = blob[0]
        y = blob[1]
        sigma = blob[2]
        brightness = blob[3]
        for i in range(-int(sigma), int(sigma) + 1):
            for j in range(-int(sigma), int(sigma) + 1):
                if check_inside(x + i, y + j, req_height, req_width):
                    dist = sqrt((i) ** 2 + (j) ** 2)
                    if dist <= sigma:
                        blob_array[int(x + i), int(y + j)] += exp(-((dist / sigma) / 2)) * brightness

    blob_image = to_image(blob_array, req_width, req_height)
    save(blob_image, 'blobs_image')

    dog_picture = new_circled_image.copy()
    draw_result = ImageDraw.Draw(dog_picture)
    # for blob in blobs_dog:
    #     x = blob[1]
    #     y = blob[0]
    #     sigma = blob[2]
    #     draw_result.point((x, y), blue)
    #     for i in range(int(sigma)):
    #         draw_result.point((x, y + i), red)
    #         draw_result.point((x, y - i), red)
    #         draw_result.point((x + i, y), red)
    #         draw_result.point((x - i, y), red)
    # save(dog_picture, 'blobs_dog')

    x, y = [np.linspace(0, req_width - 1, req_width)] * 2
    dx, dy = [c[1] - c[0] for c in (x, y)]
    lap = Laplacian(h=[dx, dy])

    circle_array = to_array(new_circled_image)
    lap_array = lap(circle_array) * 5
    lap_image = to_image(lap_array, req_width, req_height)

    save(lap_image, 'laplacian')

    # circled_array = np.empty((req_width, req_height))
    # for x in range(req_width):
    #     for y in range(req_height):
    #         circled_array[x, y] = circled_pixels[x, y][0]

    # fft_image = get_fft_image(new_image, req_height, 15, 140, -1400)
    # save(fft_image, 'fft')
    #
    # grad_image = get_gradient_image(new_image, req_width, req_height)
    # save(grad_image, 'gradient'))

    # draw_result = ImageDraw.Draw(new_image)
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
    # save(new_image, 'blobs')

    array_image = np.zeros((req_width, req_height))
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

    phash = imagehash.phash(new_circled_image)
    hash_as_str = str(phash)
    print(hash_as_str)
    # new_image = rotate(new_image, 100, 256)
    # save(new_image, 'todo5')
    # phash = imagehash.phash(new_image)
    # hash_as_str = str(phash)
    # print(hash_as_str)


def run_all():
    input_files = os.listdir(input_folder)
    for input_file in sorted(input_files)[::-1]:
        if '~' in input_file:
            continue
        process_file(input_file, False)


if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bloblist_folder, exist_ok=True)
    run_all()
