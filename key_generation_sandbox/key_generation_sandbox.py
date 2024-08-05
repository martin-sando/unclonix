#!/usr/bin/env python3
import cv2
import json
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
report_folder = output_folder + '/report'
class Blob:
    def __init__(self, coords, size, brightness=None, dct_128_8=None, color=None):
        self.coords = coords
        self.size = size
        self.brightness = brightness
        self.dct_128_8 = dct_128_8
        self.color = color
    def to_json(self):
        return json.dumps({
            "coords": self.coords,
            "size": self.size,
            "brightness": self.brightness,
            "dct_128_8": self.dct_128_8,
            "color": self.color
        })
    def log(self, file):
        file.write(self.to_json() + "\n")
    def same_dot(self, blob2):
        return self.coords[0] == blob2.coords[0] and self.coords[1] == blob2.coords[1]
    @staticmethod
    def unpack(blob_json):
        blob_dict = json.loads(blob_json)
        coords = blob_dict["coords"]
        size = blob_dict["size"]
        brightness = blob_dict["brightness"]
        dct_128_8 = blob_dict["dct_128_8"]
        color = blob_dict["color"]
        return Blob(coords, size, brightness, dct_128_8, color)

def get_blob_list(filename):
    text_read = open(filename)
    blobs=[]
    for line in text_read:
        blob = Blob.unpack(line)
        blobs.append(blob)
    return blobs
def check_inside(x, y, w, h, overflow=0, rd=0):
    return (((x - rd) >= -overflow) & ((y - rd) >= -overflow)) & (
            ((x + rd) < (w + overflow)) & ((y + rd) < (h + overflow)))


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


def find_circle_ph1(image, compression_power, saved_image):
    rmin = int(min(image.height, image.width) / 8)
    rmax = int(min(image.height, image.width) / 1.9)
    precision = 0.7
    circles = find_circles(image, rmin, rmax, precision)

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


def find_circle_ph2(image, req_size, compression_power):
    width = image.width
    height = image.height
    saved_logo = image.copy()
    image = image.resize((width // compression_power, height // compression_power))
    width = image.width
    height = image.height
    rmin = int(min(width, height) / 2.7)
    rmax = int(min(width, height) / 1.9)
    precision = 0.7
    circles = find_circles(image, rmin, rmax, precision)

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
            if (check_inside(x1 + x0, y1 + y0, saved_image.width, saved_image.height)):
                draw_result.point((x1 + r0, y1 + r0), saved_pixels[x1 + x0, y1 + y0])
            else:
                draw_result.point((x1 + r0, y1 + r0), (0, 0, 0))
    return circle_image


def trim(image, req_size, skip_factor):
    req_width = req_size[0]
    req_height = req_size[1]
    saved_pixels = image.load()
    #redrawing image to make circle big and erase blackground
    r = (req_width / 2)

    new_circled_image = Image.new("RGB", req_size)
    draw_result = ImageDraw.Draw(new_circled_image)

    for x1 in range(req_width):
        for y1 in range(req_height):
            dist = sqrt((x1 - r) ** 2 + (y1 - r) ** 2) / r
            if dist >= 1:
                draw_result.point((x1, y1), (0, 0, 0))
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
    return new_circled_image


def brighten(image, bright_coef):
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
        brightened_blobs.append(blob)
    return brightened_blobs


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


def get_field_image(input_image, width, height, precision, hru, low_b, up_b, cell, scale=True, contrast=128,
                    cutter_size=128, blobs=None, rgb=False, tournament=False):
    r = width // 2
    result_array = np.zeros((width, height, 3))
    dct_sum = np.zeros((128, 128))
    dct_sum[0, 0] = 1
    total_sum1 = 0
    total_sum2 = 0
    total_sum3 = 0
    req_dots = []
    dct_fields = []
    dct_count = 0;
    i_range = range(low_b, up_b + 1)
    j_range = range(low_b, up_b + 1)
    if cell:
        i_range = [low_b]
        j_range = [up_b]
    if blobs is None:
        for x in range(width // precision):
            for y in range(height // precision):
                req_dots.append((x * precision, y * precision))
    else:
        for blob in blobs:
            req_dots.append(blob.coords)

    for dot in req_dots:
        coord_0 = int(dot[0])
        coord_1 = int(dot[1])
        dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
        if dist > 0.8:
            dct_fields.append([[]])
            continue
        angle = atan2((coord_1 - r), (coord_0 - r))
        blob_img = input_image.crop(
            (coord_0 - cutter_size, coord_1 - cutter_size, coord_0 + cutter_size + 1, coord_1 + cutter_size + 1))
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
        for i in i_range:
            for j in j_range:
                if scale:
                    dct_sum[i, j] = dct_sum[i, j] + abs(dct_array[i, j])
                else:
                    dct_sum[i, j] = 1
        dct_count = dct_count + 1
        dct_fields.append(dct_array)
    dct_sum = dct_sum / dct_count

    for i in range(len(req_dots)):
        coord_0 = int(req_dots[i][0])
        coord_1 = int(req_dots[i][1])
        dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
        if dist > 0.8:
            continue
        dct_array = dct_fields[i]
        score = [0, 0, 0]
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 0
        if not tournament:
            for i in i_range:
                for j in j_range:
                    s1 += hru[0][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (
                                dct_array[i, j] / sqrt(dct_sum[i, j]))
                    s2 += hru[1][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (
                            dct_array[i, j] / sqrt(dct_sum[i, j]))
                    s3 += hru[2][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (
                            dct_array[i, j] / sqrt(dct_sum[i, j]))
                    s4 += hru[3][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (
                            dct_array[i, j] / sqrt(dct_sum[i, j]))
                    if rgb:
                        score[1] += hru[1][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (
                                    dct_array[i, j] / sqrt(dct_sum[i, j]))
                        score[2] += hru[2][(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (
                                    dct_array[i, j] / sqrt(dct_sum[i, j]))
            score[0] = min(s1, s2, s3, s4)
            total_sum1 += abs(score[0])
            total_sum2 += abs(score[1])
            total_sum3 += abs(score[2])
        else:
            for hru_elem in hru:
                sum = 0
                for i in i_range:
                    for j in j_range:
                        sum += hru_elem[(i - low_b) * (up_b - low_b + 1) + (j - low_b)] * (
                                    dct_array[i, j] / dct_sum[i, j])
                if sum > 0:
                    score[0] = score[0] + 30
                else:
                    break
            total_sum1 += score[0]
        result_array[coord_0, coord_1, 0] = score[0]
        result_array[coord_0, coord_1, 1] = score[1]
        result_array[coord_0, coord_1, 2] = score[2]
    field_image = input_image.copy()
    draw_result = ImageDraw.Draw(field_image)

    scaling1 = total_sum1 / (dct_count * contrast)
    scaling2 = total_sum2 / (dct_count * contrast)
    scaling3 = total_sum3 / (dct_count * contrast)

    if not (blobs is None):
        for blob in blobs:
            x = int(blob.coords[0])
            y = int(blob.coords[1])
            colors = (128 + int(result_array[x, y][0] / max(scaling1, 0.0001)),
                      128 + int(result_array[x, y][1] / max(scaling2, 0.0001)),
                      128 + int(result_array[x, y][2] / max(scaling3, 0.0001)))
            if (tournament):
                colors = (int(result_array[x, y][0]), 0, 0)
            elif not rgb:
                colors = (128 + int(result_array[x, y][0] / max(scaling1, 0.0001)), 0, 0)
            draw_result.point((x, y), colors)
            for i in range(int(blob.size)):
                draw_result.point((x, y + i), colors)
                draw_result.point((x, y - i), colors)
                draw_result.point((x + i, y), colors)
                draw_result.point((x - i, y), colors)
    else:
        for x in range(width // precision):
            for y in range(height // precision):
                for x1 in range(precision):
                    for y1 in range(precision):
                        if x1 + y1 == 0:
                            continue
                        coord_0 = x * precision + x1
                        coord_1 = y * precision + y1
                        dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
                        if dist > 0.8:
                            continue
                        comp_1 = result_array[x * precision, y * precision] * (1 - x1 / precision) * (
                                    1 - y1 / precision)
                        comp_2 = result_array[x * precision, y * precision + precision] * (1 - x1 / precision) * (
                                y1 / precision)
                        comp_3 = result_array[x * precision + precision, y * precision] * (x1 / precision) * (
                                1 - y1 / precision)
                        comp_4 = result_array[x * precision + precision, y * precision + precision] * (
                                    x1 / precision) * (
                                         y1 / precision)
                        result_array[coord_0, coord_1] = comp_1 + comp_2 + comp_3 + comp_4
        for x in range(width):
            for y in range(height):
                dist = sqrt((x - r) ** 2 + (y - r) ** 2) / r
                color = (0, 0, 0)
                if dist < 0.8:
                    color = (int(128 + result_array[x, y][0] // max(scaling1, 0.0001)),
                             int(128 + result_array[x, y][1] // max(scaling2, 0.0001)),
                             int(128 + result_array[x, y][2] // max(scaling3, 0.0001)))
                    if (tournament):
                        color = (int(result_array[x, y][0]), 0, 0)
                    elif not rgb:
                        color = (128 + int(result_array[x, y][0] / max(scaling1, 0.0001)), 0, 0)
                draw_result.point((x, y), color)

    return field_image


def add_dcts(input_image, width, height, blobs, cutter_size=128):
    calculated_blobs = []
    r = width // 2
    for blob in blobs:
        coord_0 = int(blob.coords[0])
        coord_1 = int(blob.coords[1])
        dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
        if dist > 0.8 or dist < 0.4:
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
        dct_corner = [(x.tolist())[0:8] for x in dct_array[0:8]]
        blob.dct_128_8 = dct_corner
        calculated_blobs.append(blob)
    return calculated_blobs


def add_colors(blobs, hru_array):
    colored_blobs = []
    for blob in blobs:
        dct_array = blob.dct_128_8
        colors = [0, 0, 0]
        sum = 0
        for i in range(4):
            for j in range(4):
                sum += hru_array[0][i * 8 + j] * dct_array[i][j]
        colors[0] = sum
        sum = 0
        for i in range(4):
            for j in range(4, 8):
                sum += hru_array[1][i * 8 + j] * dct_array[i][j]
        colors[1] = sum
        sum = 0
        for i in range(4, 8):
            for j in range(4):
                sum += hru_array[2][i * 8 + j] * dct_array[i][j]
        colors[2] = sum
        blob.color = colors
        colored_blobs.append(blob)
    return colored_blobs


def get_best_color(blobs, amount, color_num):
    colors = []
    for blob in blobs:
        colors.append(blob.color[color_num])
    colors.sort()
    best_blobs = []
    for blob in blobs:
        if (blob.color[color_num] >= colors[len(blobs) - amount]):
            best_blobs.append(blob)
    return best_blobs


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


def find_triangles(r, best_blobs_color, req_angles, threshold):
    triangles = []
    for blob1 in best_blobs_color[0]:
        for blob2 in best_blobs_color[1]:
            if blob1.same_dot(blob2):
                continue
            for blob3 in best_blobs_color[2]:
                if blob1.same_dot(blob3) or blob2.same_dot(blob3):
                    continue
                coord1 = (blob1.coords[0] - r, blob1.coords[1] - r)
                coord2 = (blob2.coords[0] - r, blob2.coords[1] - r)
                coord3 = (blob3.coords[0] - r, blob3.coords[1] - r)
                dist_12 = sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
                dist_13 = sqrt((coord1[0] - coord3[0]) ** 2 + (coord1[1] - coord3[1]) ** 2)
                dist_23 = sqrt((coord2[0] - coord3[0]) ** 2 + (coord2[1] - coord3[1]) ** 2)
                if (dist_12 > 300 and dist_13 > 300) and dist_23 > 300:
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
                    cur_score = abs(angles[0] - req_angles[0]) + abs(angles[1] - req_angles[1]) + abs(
                        angles[2] - req_angles[2])
                    if cur_score < threshold:
                        triangles.append((blob1, blob2, blob3))
    return triangles


def draw_triangles(image, triangles, best_blobs_color):
    draw = ImageDraw.Draw(image)
    for (blob1, blob2, blob3) in triangles:
        draw.line([(blob1.coords[0], blob1.coords[1]), (blob2.coords[0], blob2.coords[1])], fill="red", width=0)
        draw.line([(blob2.coords[0], blob2.coords[1]), (blob3.coords[0], blob3.coords[1])], fill="green", width=0)
        draw.line([(blob3.coords[0], blob3.coords[1]), (blob1.coords[0], blob1.coords[1])], fill="blue", width=0)

    for i in range(3):
        for blob in best_blobs_color[i]:
            coords = blob.coords
            color = (0, 0, 0)
            if i == 0:
                color = (255, 0, 0)
            if i == 1:
                color = (0, 255, 0)
            if i == 2:
                color = (0, 0, 255)
            draw.point(coords, color)
            for i in range(6):
                draw.point((coords[0] - i, coords[1]), color)
                draw.point((coords[0], coords[1] - i), color)
                draw.point((coords[0] + i, coords[1]), color)
                draw.point((coords[0], coords[1] + i), color)
    return image


def filter_image(image, bound1, bound2):
    filtered_image = Image.new("RGB", image.size())
    draw_result = ImageDraw.Draw(filtered_image)
    saved_pixels = image.load()
    for x1 in range(image.width):
        for y1 in range(image.height):
            if (saved_pixels[x1, y1][0] > bound1):
                draw_result.point((x1, y1), (255, 255, 255))
            elif (saved_pixels[x1, y1][0] > bound2):
                for x2 in range(-10, 10):
                    for y2 in range(-10, 10):
                        if (saved_pixels[x1 + x2, y1 + y2][0] > 250):
                            draw_result.point((x1, y1), (255, 255, 255))
    return filtered_image


def get_blob_info(image, coords, blobs, mask_num, filename, hru, bound_1, bound_2, cutter_size=128):
    label_folder = report_folder + "/" + str(mask_num)
    os.makedirs(label_folder, exist_ok=True)
    text_file = open(os.path.join(label_folder, filename + '.txt'), 'w')

    def log(text, end="\n"):
        text_file.write(text + end)

    def log_picture(image, tag):
        image.save(os.path.join(label_folder, filename + "_" + tag + ".png"))

    def log_vector(vec):
        log("[", end='')
        for elem in vec:
            s = str(elem)
            if (len(s) > 6):
                s = s[:6]
            while (len(s) < 6):
                s = s + '0'
            log(s, end='\t')
        log("]", end='\n')

    def log_matrix(mtr):
        log("[", end='\n')
        for vec in mtr:
            log_vector(vec)
        log("]", end='\n')

    closest_blob = (0, 0, 0)
    closest_dist = 10000
    for blob in blobs:
        dist = (coords[0] - blob.coords[0]) ** 2 + (coords[1] - blob.coords[1]) ** 2
        if dist < closest_dist:
            closest_dist = dist
            closest_blob = blob
    log("Closest blob found to (" + str(coords[0]) + ", " + str(coords[1]) + ") is (" + str(
        closest_blob.coords[0]) + ", " + str(closest_blob.coords[1]) + ")")

    draw_result = ImageDraw.Draw(image);
    draw_result.point(closest_blob.coords, (0, 0, 255))
    x = int(closest_blob.coords[0])
    y = int(closest_blob.coords[1])
    draw_result.point((x, y), (0, 0, 255))
    for i in range(int(closest_blob.size)):
        draw_result.point((x, y + i), (0, 0, 255))
        draw_result.point((x, y - i), (0, 0, 255))
        draw_result.point((x + i, y), (0, 0, 255))
        draw_result.point((x - i, y), (0, 0, 255))
    r = image.width // 2
    angle = atan2((y - r), (x - r))
    blob_img = image.crop(
        (x - cutter_size, y - cutter_size, x + cutter_size,
         y + cutter_size))
    rot_image = blob_img.rotate((angle * (180 / pi)))

    blob_img = rot_image.crop(
        (int(cutter_size / 2) + 1, int(cutter_size / 2) + 1, int(cutter_size * (3 / 2)) + 1, int(cutter_size * (3 / 2)) + 1))
    log_picture(blob_img, "around")
    blob_pixels = blob_img.load()
    array_image = np.zeros((cutter_size, cutter_size))
    for x1 in range(cutter_size):
        for y1 in range(cutter_size):
            array_image[x1, y1] = blob_pixels[x1, y1][0] + 0.0
    dct_array = cv2.dct(array_image)
    log("Its dct looks like (before normalisation): ")
    log_matrix(dct_array)
    log("hru is")
    log_vector(hru)
    log("Unnormalised, its elems from " + str(bound_1) + " to " + str(bound_2) + ":")
    dct_hru = np.zeros((bound_2 - bound_1 + 1, bound_2 - bound_1 + 1))
    for i in range(bound_1, bound_2 + 1):
        for j in range(bound_1, bound_2 + 1):
            dct_hru[i - bound_1, j - bound_1] += hru[(i - bound_1) * (bound_2 - bound_1 + 1) + (j - bound_1)] * (
            dct_array[i, j])
    log_matrix(dct_hru)


def process_file(input_file, full_research_mode, mask):
    random.seed(566)
    hru_array = []
    for i in range(20):
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
    r = req_width // 2

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

    input_image = return_grayscale(input_image, width, height)

    save(input_image, 'to_gray')

    input_image = input_image.resize((width // compression_power, height // compression_power))
    save(input_image, 'compressed')

    logo_image = find_circle_ph1(input_image, compression_power, saved_image)
    save(logo_image, 'large_circle')

    compression_power = logo_image.width // 200
    circled_image = find_circle_ph2(logo_image, req_size, compression_power)

    circled_image = circled_image.copy()
    saved_circle_image = circled_image.copy()
    save(circled_image, 'circled')

    new_circled_image = trim(circled_image, req_size, skip_factor=(circled_image.width // 150))
    save(new_circled_image, 'trimmed')

    new_circled_image = brighten(new_circled_image, bright_coef=15)
    save(new_circled_image, 'brightened')

    x, y = [np.linspace(0, req_width - 1, req_width)] * 2
    dx, dy = [c[1] - c[0] for c in (x, y)]
    lap = Laplacian(h=[dx, dy])

    circle_array = to_array(new_circled_image)
    lap_array = lap(circle_array) * 5
    lap_image = to_image(lap_array, req_width, req_height)

    save(lap_image, 'laplacian')

    new_circled_image = new_circled_image.copy()
    circled_pixels = new_circled_image.load()

    morph_image = rgb2gray(new_circled_image)
    blobs_log = blob_log(morph_image, min_sigma=req_width / 450, max_sigma=req_width / 190, num_sigma=10, threshold=.03,
                         overlap=0.5)
    blobs_log[:, 2] = (blobs_log[:, 2] * np.sqrt(2)) + 1

    blobs_obj = []

    for blob in blobs_log:
        coords = [blob[1], blob[0]]
        size = blob[2]
        blobs_obj.append(Blob(coords, size))

    blobs_obj = brighten_blobs(new_circled_image, blobs_obj)
    blobs_obj = add_dcts(new_circled_image, req_width, req_height, blobs_obj)
    blobs_obj = add_colors(blobs_obj, hru_array)
    text_file = open(os.path.join(report_folder, mask, filename + '.txt'), 'w')
    for blob in blobs_obj:
        blob.log(text_file)
    blobs_obj = get_blob_list(os.path.join(report_folder, mask, filename + '.txt'))
    red_blobs = get_best_color(blobs_obj, 50, 0)
    green_blobs = get_best_color(blobs_obj, 50, 1)
    blue_blobs = get_best_color(blobs_obj, 50, 2)
    colors_blobs = [red_blobs, green_blobs, blue_blobs]
    triangles = find_triangles(req_width // 2, colors_blobs, (60, 60, 60), 2)
    copy_image = new_circled_image.copy()
    triangles_image = draw_triangles(copy_image, triangles, colors_blobs)
    save(triangles_image, 'chaos')
    ######################################################################
    # Previous steps are converting initial image to full-image label grayscale with good brightness
    # They won't change much
    # Below are experiments
    ######################################################################

    ######################################################################
    # get_field_image_manual (sort of):
    #
    # blobs - if specified, list of blobs to calculate fields in
    # if None or not specified, field will be calculated over entire image
    #
    # precision - if (blobs=None), determines how precisely should field be generated (lower = more precise, up to 1)
    #
    # cell - if False, use 2-dimensional slice of dct [low_b..up_b] * [low_b..up_b], multiplied by hru
    # if True, use dot of dct [low_b, up_b]
    #
    # scale - should dct_elems be rescaled based on average
    #
    # contrast - determines how bright image will be (the more the brighter, recommended range(30..150)) [note: tournaments unsupported]
    #
    # rgb - if True, 3 color components would be generated using 3 hru's
    # if False, only red color will be generated
    #
    # tournament - if true, color dots depending on how many hru's in a row resulted in positive score (only if rgb=False)
    #
    # You may use playground below to see (and also report bugs)
    ######################################################################

    pairs = [(3, 5)]
    dots = [(6, 2)]
    for pair in pairs:
        field_image = get_field_image(new_circled_image, req_width, req_height, precision=10, contrast=70,
                                      hru=hru_array, low_b=pair[0], up_b=pair[1], cell=False, scale=True, blobs=None,
                                      rgb=True, tournament=False)
        save(field_image, 'field_image_array' + str(pair[0]) + '..' + str(pair[1]))

    for dot in dots:
        field_image = get_field_image(new_circled_image, req_width, req_height, precision=10, contrast=70,
                                      hru=hru_array, low_b=dot[0], up_b=dot[1], cell=True, scale=True, blobs=None,
                                      rgb=True, tournament=False)
        save(field_image, 'field_image_dot' + str(dot[0]) + '.' + str(dot[1]))

    field_image = get_field_image(new_circled_image, req_width, req_height, 10, hru_array, 0, 3, contrast=70, cell=False, scale=True,
                                  blobs=blobs_obj, rgb=False, tournament=False)
    save(field_image, 'field_image_blob' + '0' + '..' + '5')
    field_image = get_field_image(new_circled_image, req_width, req_height, 10, hru_array, 0, 3, contrast=70, cell=False, scale=True,
                                  blobs=None, rgb=False, tournament=False)
    save(field_image, 'field_image_array' + '0' + '..' + '5')
    #some sode using my filter, only works well if tournament=False, rgb=False and blobs=None
    # angle_image = get_angle_image(field_image, req_width, req_height, mode="both", precision=5)
    # save(angle_image, 'angle_image_array' + '0' + '..' + '5')
    #
    #
    # filtered_image = filter_image(angle_image, 250, 200)
    # save(filtered_image, 'fitlered')
    #
    # morph_image = rgb2gray(angle_image)
    # blobs_log = blob_log(morph_image, min_sigma=1, max_sigma=req_width / 190, num_sigma=10, threshold=.10,
    #                      overlap=0.5)
    # blobs_log[:, 2] = (blobs_log[:, 2] * np.sqrt(2)) + 1
    #
    # dot_image = angle_image.copy()
    # draw_result = ImageDraw.Draw(dot_image)
    # transformed_blobs = []
    #
    # for blob in blobs_log:
    #     transformed_blobs.append((blob[1], blob[0], blob[2]))
    # blobs_log = transformed_blobs
    # for blob in blobs_log:
    #     x = blob[0]
    #     y = blob[1]
    #     sigma = blob[2]
    #     draw_result.point((x, y), blue)
    #     for i in range(int(sigma)):
    #         draw_result.point((x, y + i), blue)
    #         draw_result.point((x, y - i), blue)
    #         draw_result.point((x + i, y), blue)
    #         draw_result.point((x - i, y), blue)
    #
    # save(dot_image, "dots")
    #
    # #find best_fitting triangles using blobs
    # triangle_image = find_triangle(angle_image, blobs_log, (55, 60, 65))
    # save(triangle_image, "triangle")

    if not full_research_mode:
        return


    # blobs_dog = blob_dog(morph_image, min_sigma=1.2, max_sigma=req_width / 170, threshold=.03)
    # blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
    # blobs_doh = blob_doh(morph_image, max_sigma=20, threshold=.01)

    log_picture = new_circled_image.copy()
    draw_result = ImageDraw.Draw(log_picture)

    draw_result_bright = ImageDraw.Draw(saved_circle_image)

    brightened_blobs = brighten_blobs(log_picture, blobs_log)

    blob_array = np.zeros((req_width, req_height))
    for blob in blobs_obj:
        x = blob.coords[0]
        y = blob.coords[1]
        sigma = blob.size
        brightness = blob.brightness
        for i in range(-int(sigma), int(sigma) + 1):
            for j in range(-int(sigma), int(sigma) + 1):
                if check_inside(x + i, y + j, req_height, req_width):
                    dist = sqrt((i) ** 2 + (j) ** 2)
                    if dist <= sigma:
                        blob_array[int(x + i), int(y + j)] += exp(-((dist / sigma) / 2)) * brightness

    blob_image = to_image(blob_array, req_width, req_height)
    save(blob_image, 'blobs_image')

    x, y = [np.linspace(0, req_width - 1, req_width)] * 2
    dx, dy = [c[1] - c[0] for c in (x, y)]
    lap = Laplacian(h=[dx, dy])

    circle_array = to_array(new_circled_image)
    lap_array = lap(circle_array) * 5
    lap_image = to_image(lap_array, req_width, req_height)

    save(lap_image, 'laplacian')

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


def run_all(mask):
    input_files = os.listdir(input_folder)
    for input_file in sorted(input_files)[::-1]:
        if '~' in input_file or mask not in input_file:
            continue
        process_file(input_file, False, mask)


if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bloblist_folder, exist_ok=True)
    os.makedirs(report_folder, exist_ok=True)
    mask = ''
    for arg in sys.argv:
        if arg.startswith('--mask='):
            mask = arg[7:]
    run_all(mask)
