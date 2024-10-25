#!/usr/bin/env python3
import cv2
import random
from PIL import Image, ImageDraw
import imagehash
from math import sqrt, pi, cos, sin, exp, atan2
from canny import find_circles, rotate, compute_gradient
from collections import defaultdict
import sys
import os.path
import numpy as np
from findiff import Gradient, Divergence, Laplacian, Curl
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from Blob import Blob
import utils

check_inside = utils.check_inside
black, blue, red, green, white = utils.black, utils.blue, utils.red, utils.green, utils.white
req_size, req_width, req_height, r = utils.req_size, utils.req_width, utils.req_height, utils.r
to_array, to_image = utils.to_array, utils.to_image
save = utils.save
run_experiment = utils.run_experiment
def finding_circle_ph1(image, compression_power, saved_image):
    rmin = int(min(image.height, image.width) / 5)
    rmax = int(min(image.height, image.width) / 1.9)
    precision = 0.7
    circles = find_circles(image, rmin, rmax, precision, 13)

    if not circles:
        print('Error: not found circles')
        return

    max_overflow = 15
    x0, y0, r0, pr0 = get_best_circle(circles, image.width, image.height, max_overflow)

    if r0 == 0:
        print('Error: r0 = 0')
        return

    logo_image = get_circle_image(x0, y0, int(r0 * compression_power * 1.1), compression_power, saved_image)
    return logo_image


def finding_circle_ph2(image, compression_power):
    width = image.width
    height = image.height
    saved_logo = image.copy()
    image = image.resize((width // compression_power, height // compression_power))
    width = image.width
    height = image.height
    rmin = int(min(width, height) / 2.7)
    rmax = int(min(width, height) / 1.9)
    precision = 0.5
    circles = find_circles(image, rmin, rmax, precision, 27)

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
                    draw_result.point((x1 + r0, y1 + r0), (0, 0, 0))
                else:
                    draw_result.point((x1 + r0, y1 + r0), saved_pixels[x1 + x0, y1 + y0])

    circled_image = circled_image.resize(req_size)
    return circled_image


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
            if check_inside(x1 + x0, y1 + y0, saved_image.width, saved_image.height):
                draw_result.point((x1 + r0, y1 + r0), saved_pixels[x1 + x0, y1 + y0])
            else:
                draw_result.point((x1 + r0, y1 + r0), black)
    return circle_image


def trimming(image, skip_factor, precision):
    saved_pixels = image.load()
    #redrawing image to make circle big and erase blackground

    new_circled_image = Image.new("RGB", req_size)
    draw_result = ImageDraw.Draw(new_circled_image)

    median_field = np.zeros((req_width, req_height, 3))

    for x1 in range(req_width):
        for y1 in range(req_height):
            dist = sqrt((x1 - r) ** 2 + (y1 - r) ** 2) / r
            if dist >= 1:
                continue
            else:
                if dist < 0.97 and (x1 % precision != 0 or y1 % precision != 0):
                    continue
                surrounding_colours = [[], [], []]
                close_r = int((req_width * 0.03) / skip_factor)
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
                median = [surrounding_colours[0][len(surrounding_colours[0]) // 2],
                          surrounding_colours[1][len(surrounding_colours[1]) // 2],
                          surrounding_colours[2][len(surrounding_colours[2]) // 2]]
                median_field[x1, y1] = median

    for x1 in range(req_width):
        for y1 in range(req_height):
            dist = sqrt((x1 - r) ** 2 + (y1 - r) ** 2) / r
            if dist >= 0.93:
                draw_result.point((x1, y1), black)
            else:
                if median_field[x1, y1][0] == 0:
                    lx = (x1 // precision) * precision
                    rx = lx + precision
                    ly = (y1 // precision) * precision
                    ry = ly + precision
                    comp_1 = median_field[lx, ly] * (1 - (x1 % precision) / precision) * (
                            1 - (y1 % precision) / precision)
                    comp_2 = median_field[lx, ry] * (1 - (x1 % precision) / precision) * (
                            (y1 % precision) / precision)
                    comp_3 = median_field[rx, ly] * ((x1 % precision) / precision) * (
                            1 - (y1 % precision) / precision)
                    comp_4 = median_field[rx, ry] * ((x1 % precision) / precision) * (
                            (y1 % precision) / precision)
                    median_field[x1, y1] = comp_1 + comp_2 + comp_3 + comp_4
                color = int(abs(saved_pixels[x1, y1][0] - median_field[x1, y1][0]) +
                            abs(saved_pixels[x1, y1][1] - median_field[x1, y1][1]) +
                            abs(saved_pixels[x1, y1][2] - median_field[x1, y1][2])) // 3
                draw_result.point((x1, y1), (color, color, color))
    return new_circled_image


def brightening(image, bright_coef):
    pixels = image.load()
    draw_result = ImageDraw.Draw(image)

    total_brightness = 0
    for x1 in range(image.width):
        for y1 in range(image.height):
            total_brightness += pixels[x1, y1][0]

    req_brightness = bright_coef * image.width * image.height
    brightness_coef = req_brightness / total_brightness
    for x1 in range(image.width):
        for y1 in range(image.height):
            color = int(pixels[x1, y1][0] * brightness_coef)
            draw_result.point((x1, y1), (color, color, color))
    return image

def _processed(image, power):
    pixels = image.load()
    draw_result = ImageDraw.Draw(image)
    deltas = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            dist = dx**2 + dy**2
            if dist > 9:
                continue
            deltas.append((dx, dy))
    for x1 in range(image.width):
        for y1 in range(image.height):
            if x1 < 2 or x1 > image.width - 3:
                draw_result.point((x1, y1), black)
                continue
            if y1 < 2 or y1 > image.width - 3:
                draw_result.point((x1, y1), black)
                continue
            total_brightness = 0
            for (dx, dy) in deltas:
                total_brightness += pixels[x1 + dx, y1 + dy][0]
            total_brightness /= len(deltas)
            color = int(255 * max(total_brightness - power, 0) / (255 - power))
            draw_result.point((x1, y1), (color, color, color))
    return image

def brighten_blobs(image, blobs):
    pixels = image.load()
    brightened_blobs = []
    for blob in blobs:
        x = blob.coords[0]
        y = blob.coords[1]
        size = blob.size
        brightness = 0
        for i in range(-int(size), int(size) + 1):
            for j in range(-int(size), int(size) + 1):
                if check_inside(x + i, y + j, image.height, image.width):
                    dist = sqrt((i) ** 2 + (j) ** 2)
                    if dist <= size ** 2:
                        brightness += exp(-(dist / (2 * size))) * pixels[(x, y)][0]
        brightness = brightness / (2 * size * size)
        blob.brightness = brightness
        #if (brightness > 180 and brightness * size > 1000):
        brightened_blobs.append(blob)
    return brightened_blobs


def get_fft_image(image, image_size, fft_radius, coef_1, coef_2):
    array_image = utils.to_array(image)

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

    array_img = utils.to_array(image)
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


def get_partial_image(image, req_width, req_height):
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
                draw_parts.point((i, j), black)
    return parts_image

def add_dcts(input_image, width, height, blobs, cutter_size=128):
    calculated_blobs = []
    r = width // 2
    for blob in blobs:
        dist = sqrt((blob.coords[0] - r) ** 2 + (blob.coords[1] - r) ** 2) / r
        blob_img = utils.get_rotated_surroundings(input_image, blob.coords, cutter_size)
        blob_pixels = blob_img.load()

        array_image = np.zeros((cutter_size, cutter_size))
        for x1 in range(cutter_size):
            for y1 in range(cutter_size):
                array_image[x1, y1] = blob_pixels[x1, y1][0] + 0.0

        dct_array = cv2.dct(array_image)
        dct_corner = [(x.tolist())[0:8] for x in dct_array[0:8]]
        blob.dct_128_8 = dct_corner
        bmp_128_7 = to_array(blob_img.resize((7, 7)))
        bmp_128_7_t = []
        for i in range(7):
            lst = []
            for j in range(7):
                lst.append(bmp_128_7[j, i])
            bmp_128_7_t.append(lst)
        bmp_128_15 = to_array(blob_img.resize((15, 15)))
        blob.bmp_128_7 = bmp_128_7_t
        bmp_128_15_t = []
        for i in range(15):
            lst = []
            for j in range(15):
                lst.append(bmp_128_15[j, i])
            bmp_128_15_t.append(lst)
        blob.bmp_128_15 = bmp_128_15_t
        calculated_blobs.append(blob)
    return calculated_blobs


def compressing(image, compression_power):
    return image.resize((image.width // compression_power, image.height // compression_power))

def blurring(image, kernel_size):
    circled_array = to_array(image)
    circled_array = cv2.GaussianBlur(circled_array, kernel_size,0)
    image = to_image(circled_array, image.width, image.height)
    return image

def binarying(image, threshold1, threshold2):
    draw_result = ImageDraw.Draw(image)
    pixels = image.load()
    for x1 in range(req_width):
        for y1 in range(req_height):
            if pixels[x1, y1][0] < threshold1:
                draw_result.point((x1, y1), black)
            else:
                t = False
                for dx in range(-5, 5):
                    for dy in range(-5, 5):
                        if check_inside(x1+dx, y1+dy, req_width, req_height):
                            if pixels[x1+dx, y1+dy][0] >= threshold2:
                                t = True
                                break
                    if t:
                        break
                if not t:
                    draw_result.point((x1, y1), black)
    return image

def logging_blobs(image, filename):
    image = image.copy()
    morph_image = rgb2gray(image)
    blobs_log = blob_log(morph_image, min_sigma=req_width / 750, max_sigma=req_width / 120, num_sigma=10, threshold=.35,
                         overlap=0.5)
    blobs_log[:, 2] = (blobs_log[:, 2] * np.sqrt(2)) + 1
    blobs_obj = []

    for blob in blobs_log:
        coords = [blob[1], blob[0]]
        size = blob[2]
        blobs_obj.append(Blob(coords, size))

    blobs_obj = brighten_blobs(image, blobs_obj)
    blobs_obj = add_dcts(image, req_width, req_height, blobs_obj)
    text_file = open(os.path.join(utils.bloblist_folder, filename + '.txt'), 'w')
    color_blobs = {}
    for blob in blobs_obj:
        blob.log(text_file)
        color_blobs[blob] = blue
    image = utils.draw_blobs(image, 'plus', blobs_obj, color_blobs)
    return image

def laplacian(image):
    x, y = [np.linspace(0, req_width - 1, req_width)] * 2
    dx, dy = [c[1] - c[0] for c in (x, y)]
    lap = Laplacian(h=[dx, dy])
    circle_array = to_array(image)
    lap_array = lap(circle_array) * 5
    lap_image = to_image(lap_array, req_width, req_height)
    return lap_image

def dilate(image, power):
    img_2 = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    if power > 0:
        image = cv2.dilate(img_2, None, iterations=power)
    else:
        image = cv2.erode(img_2, None, iterations=-power)
    return utils.transpose(utils.to_image(image, req_width, req_height))


def process_photo(input_file, full_research_mode, directory=utils.input_folder):
    filename = input_file.split('.')[0]
    utils.set_file_name(filename)
    utils.set_picture_number(0)
    utils.set_phase_time(1)
    utils.set_save_subfolder('')
    print('Processing ' + filename)

    image = Image.open(os.path.join(directory, input_file))
    saved_image = image.copy()

    compression_power = image.width // 100

    save(image, 'input')

    image = run_experiment(compressing, image, compression_power)

    image = run_experiment(finding_circle_ph1, image, compression_power, saved_image)

    image = run_experiment(finding_circle_ph2, image, image.width // 200)

    image = run_experiment(trimming, image, (image.width // 150), 10)

    image = run_experiment(brightening, image, 15)

    utils.set_save_subfolder('report')
    run_experiment(laplacian, image)
    utils.set_save_subfolder('')

    image = run_experiment(blurring, image, (15, 15))

    image = run_experiment(binarying, image, 70, 130)

    utils.set_picture_number(utils.image_processing_picture_number_result)

    #image = run_experiment(dilate, image, -2)
    image = run_experiment(_processed, image, 12)

    utils.set_save_subfolder('report')
    run_experiment(laplacian, image)

    utils.set_save_subfolder('')
    run_experiment(logging_blobs, image, filename)