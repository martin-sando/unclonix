#!/usr/bin/env python
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector, return_grayscale
from collections import defaultdict
import sys
import os.path

input_folder = '../input'
output_folder = '../output'

def process_file(input_file):
    filename = input_file.split('.')[0]
    print('Processing ' + filename)
    white = (255, 255, 255)
    black = (0, 0, 0)
    req_width = 1024
    req_length = 1024
    req_size = (req_width, req_length)

    input_image = Image.open(os.path.join(input_folder, input_file))
    log_picture_number = 0
    def new_log_picture():
        nonlocal log_picture_number
        log_picture_number += 1
        return os.path.join(output_folder, filename + "_" + str(log_picture_number) + ".png")

    width, height = input_image.width, input_image.height
    compression_power = width // 100
    #to_gray
    input_image = return_grayscale(input_image, width, height)

    saved_gray_image = input_image.copy()

    input_image.save(new_log_picture())

    #compress image copy to effectively find a circle
    input_image = input_image.resize((width // compression_power, height // compression_power))
    input_image.save(new_log_picture())

    width, height = input_image.width, input_image.height

    input_pixels = input_image.load()


    # Find circles, assuming their D is at least half of min(h, w)
    rmin = min(input_image.height, input_image.width)//4
    rmax = min(input_image.height, input_image.width)//2
    steps = 100
    threshold = 0.2

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x, y in canny_edge_detector(input_image):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            print(v / steps, x, y, r)
            circles.append((x, y, r))

    if not circles:
        print('Error: not found circles')
        return

    x0, y0, r0 = 0, 0, 0

    # looking for the largest circle fitting inside image
    for circle in circles:
        x = circle[0]
        y = circle[1]
        r = circle[2]
        if ((r > r0) & (((x - r) > 0) & ((y - r) > 0))) & (((x + r) < width) & ((y + r) < height)):
            x0, y0, r0 = x, y, r

    if r0 == 0:
        print('Error: r0 = 0')
        return

    #extrapolating found circle to original scale
    x0, y0, r0 = x0 * compression_power + compression_power // 2, y0 * compression_power + compression_power // 2, r0 * compression_power - compression_power // 2
    saved_gray_pixels = saved_gray_image.load()

    inner_colours = []
    outer_colours = []

    for x1 in range(-r0, r0+1):
        for y1 in range(-r0, r0+1):
            if (x1) ** 2 + (y1)**2 <= r0**2:
                inner_colours.append(saved_gray_pixels[x0 + x1, y0 + y1][0])

    inner_colours.sort()

    median = inner_colours[len(inner_colours) // 2]
    circled_image = Image.new("RGB", [2*r0+1, 2*r0+1])
    circled_image.paste(input_image)
    draw_result = ImageDraw.Draw(circled_image)

    #redrawing image to make circle big and erase blackground
    for x1 in range(-r0, r0+1):
        for y1 in range(-r0, r0+1):
            dist = sqrt((x1 ** 2 + y1 **2) / (r0 ** 2))
            if dist < 0.98:
                draw_result.point((x1 + r0, y1 + r0), saved_gray_pixels[x1 + x0, y1 + y0])
            elif dist >= 1:
                draw_result.point((x1 + r0, y1 + r0), (median, median, median))
            else:
                color = (int)(saved_gray_pixels[x1 + x0, y1 + y0][0] * (50.0 - 50 * dist) + median * (50.0 * dist - 49))
                draw_result.point((x1 + r0, y1 + r0),  (color, color, color))

    circled_image.save(new_log_picture())

    #resize to 1024*1024
    circled_image = circled_image.resize(req_size)
    circled_image.save(new_log_picture())


    #making deviant-colored points black, other - white
    binary_image = Image.new("RGB", req_size)
    output_pixels = circled_image.load()

    draw_result = ImageDraw.Draw(binary_image)

    pixel_differ = 25

    for x1 in range(req_width):
        for y1 in range(req_length):
            if abs(output_pixels[x1, y1][0] - median) > pixel_differ:
                draw_result.point((x1, y1), black)
            else:
                draw_result.point((x1, y1), white)

    binary_image.save(new_log_picture())

def run_all():
	input_files = os.listdir(input_folder)
	for input_file in input_files:
		process_file(input_file)

if __name__ == '__main__':
	os.makedirs(output_folder, exist_ok=True)
	run_all()
