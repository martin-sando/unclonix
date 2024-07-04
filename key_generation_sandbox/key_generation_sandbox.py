#!/usr/bin/env python3
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import return_grayscale, find_circles, rotate
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
    saved_image = input_image.copy()

    log_picture_number = 0

    def new_log_picture():
        nonlocal log_picture_number
        log_picture_number += 1
        return os.path.join(output_folder, filename + "_" + str(log_picture_number) + ".png")

    def tag_picture(tag):
        return os.path.join(output_folder, "_" + tag + "_" + filename + ".png")

    width, height = input_image.width, input_image.height
    compression_power = width // 100
    #to_gray
    input_image = return_grayscale(input_image, width, height)


    input_image.save(new_log_picture())

    #compress image copy to effectively find a circle
    input_image = input_image.resize((width // compression_power, height // compression_power))
    input_image.save(new_log_picture())

    width, height = input_image.width, input_image.height

    input_pixels = input_image.load()

    # Find circles, assuming their D is at least half of min(h, w)
    rmin = min(input_image.height, input_image.width) // 8
    rmax = int(min(input_image.height, input_image.width) / 1.9)
    precision = 0.7
    circles = find_circles(input_image, rmin, rmax, precision)

    if not circles:
        print('Error: not found circles')
        return

    x0, y0, r0, pr0 = 0, 0, 0, 0
    max_overflow = 15

    def check_inside(x, y, w, h, overflow=0, rd=0):
        return (((x - rd) > -overflow) & ((y - rd) > -overflow)) & (
                ((x + rd) < (w + overflow)) & ((y + rd) < (h + overflow)))

    # looking for the largest circle fitting inside image
    for circle in circles:
        x = circle[0]
        y = circle[1]
        r = circle[2]
        pr = circle[3]
        if ((pr > pr0) | ((pr == pr0) & (r > r0))) & check_inside(x, y, width, height, max_overflow, r):
            x0, y0, r0, pr0 = x, y, r, pr

    if r0 == 0:
        print('Error: r0 = 0')
        return

    #extrapolating found circle to original scale
    x0, y0, r0 = x0 * compression_power + compression_power // 2, y0 * compression_power + compression_power // 2, int(r0 * compression_power - compression_power * 2)
    width = saved_image.width
    height = saved_image.height
    saved_pixels = saved_image.load()
    logo_image = Image.new("RGB", [2 * r0 + 1, 2 * r0 + 1])
    draw_result = ImageDraw.Draw(logo_image)
    for x1 in range(-r0, r0 + 1):
        for y1 in range(-r0, r0 + 1):
            if ((x1) ** 2 + (y1) ** 2 <= r0 ** 2) & check_inside(x0 + x1, y0 + y1, width, height):
                draw_result.point((x1 + r0, y1 + r0), saved_pixels[x1 + x0, y1 + y0])
            else:
                draw_result.point((x1 + r0, y1 + r0), black)




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
    circled_image.paste(input_image)
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



    circled_image.save(new_log_picture())

    #resize to 1024*1024
    circled_image = circled_image.resize(req_size)
    circled_image.save(new_log_picture())
    circled_image.save(tag_picture("last"))

def run_all():
    input_files = os.listdir(input_folder)
    for input_file in input_files:
        process_file(input_file)

if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    run_all()