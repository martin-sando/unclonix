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
    rmin = min(input_image.height, input_image.width) // 4
    rmax = int(min(input_image.height, input_image.width) / 1.9)
    precision = 0.45
    circles = find_circles(input_image, rmin, rmax, precision)

    if not circles:
        print('Error: not found circles')
        return

    x0, y0, r0 = 0, 0, 0
    max_overflow = 15

    def check_inside(x, y, w, h, overflow=0, rd=0):
        return (((x - rd) > -overflow) & ((y - rd) > -overflow)) & (
                ((x + rd) < (w + overflow)) & ((y + rd) < (h + overflow)))

    # looking for the largest circle fitting inside image
    for circle in circles:
        x = circle[0]
        y = circle[1]
        r = circle[2]
        if (r > r0) & check_inside(x, y, width, height, max_overflow, r):
            x0, y0, r0 = x, y, r

    if r0 == 0:
        print('Error: r0 = 0')
        return

    #extrapolating found circle to original scale
    x0, y0, r0 = x0 * compression_power + compression_power // 2, y0 * compression_power + compression_power // 2, r0 * compression_power - compression_power * 2
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

    logo_image.save(new_log_picture())
    saved_pixels = logo_image.load()

    #next up: rotation
    logo_r = int(r0 * 0.95)
    colors = []
    steps = 100
    count_0 = 0

    for t in range(steps):
        pixel = saved_pixels[r0 + int(logo_r * cos(2 * pi * t / steps)), r0 + int(logo_r * sin(2 * pi * t / steps))]
        pixel_brightness = pixel[0] + pixel[1] + pixel[2]
        if pixel_brightness > (255 * 3) / 2:
            colors.append(0)
            count_0 += 1
        else:
            colors.append(1)
    pop_color = 0
    if count_0 < steps/2:
        pop_color = 1

    max_counter = 0
    max_i = 0
    for i in range(steps // 2):
        i_counter = 0
        max_i_counter = 0
        for s in range(steps // 8):
            s_color = colors[(i + s + steps) % steps]
            i_delta = 0
            if s_color != pop_color:
                i_delta += 1
            else:
                i_delta -= 1
            s_color = colors[(i - s + steps) % steps]
            if s_color != pop_color:
                i_delta += 1
            else:
                i_delta -= 1
            s_color = colors[(i + s + steps + steps // 2) % steps]
            if s_color != pop_color:
                i_delta += 1
            else:
                i_delta -= 1
            s_color = colors[(i - s + steps + steps // 2) % steps]
            if s_color != pop_color:
                i_delta += 1
            else:
                i_delta -= 1
            if i_delta > 0:
                i_counter += i_delta * i_delta
            else:
                i_counter -= i_delta * i_delta

            if i_counter > max_i_counter:
                max_i_counter = i_counter
        if max_i_counter > max_counter:
            max_counter = max_i_counter
            max_i = i
    rotation = (2 * pi * max_i) / steps

    rotated_image = rotate(logo_image, rotation, r0)
    rotated_image.save(new_log_picture())


    width, height = rotated_image.width, rotated_image.height
    compression_power = width // 100

    rotated_image = return_grayscale(rotated_image, width, height)
    rotated_image.save(new_log_picture())
    saved_rotated_image = rotated_image.copy()

    rotated_image = rotated_image.resize((width // compression_power, height // compression_power))
    rotated_image.save(new_log_picture())

    width, height = rotated_image.width, rotated_image.height

    input_pixels = rotated_image.load()

    # Find circle after rotation
    rmin = int(width / 7)
    rmax = int(width / 4)
    precision = 0.55
    circles = find_circles(rotated_image, rmin, rmax, precision)
    r0 = 0
    for circle in circles:
        x = circle[0]
        y = circle[1]
        r = circle[2]
        if (r > r0) & check_inside(x, y, width, height, max_overflow, r):
            x0, y0, r0 = x, y, r

    if r0 == 0:
        print('Error: r0 = 0')
        return

    #extrapolating found circle to original scale
    x0, y0, r0 = x0 * compression_power + compression_power // 2, y0 * compression_power + compression_power // 2, r0 * compression_power - compression_power * 2
    width = saved_rotated_image.width
    height = saved_rotated_image.height
    saved_pixels = saved_rotated_image.load()
    label_image = Image.new("RGB", [2 * r0 + 1, 2 * r0 + 1])
    draw_result = ImageDraw.Draw(label_image)
    for x1 in range(-r0, r0 + 1):
        for y1 in range(-r0, r0 + 1):
            if ((x1) ** 2 + (y1) ** 2 <= r0 ** 2) & check_inside(x0 + x1, y0 + y1, width, height):
                draw_result.point((x1 + r0, y1 + r0), saved_pixels[x1 + x0, y1 + y0])
            else:
                draw_result.point((x1 + r0, y1 + r0), black)

    label_image.save(new_log_picture())
    saved_pixels = label_image.load()

    inner_colours = [[], [], []]

    for x1 in range(-r0, r0 + 1):
        for y1 in range(-r0, r0 + 1):
            if ((x1) ** 2 + (y1) ** 2 <= r0 ** 2) & check_inside(r0 + x1, r0 + y1, r0, r0):
                inner_colours[0].append(saved_pixels[r0 + x1, r0 + y1][0])
                inner_colours[1].append(saved_pixels[r0 + x1, r0 + y1][1])
                inner_colours[2].append(saved_pixels[r0 + x1, r0 + y1][2])

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
                color = (abs(median[0] - saved_pixels[x1 + r0, y1 + r0][0]) +
                         abs(median[1] - saved_pixels[x1 + r0, y1 + r0][1]) +
                         abs(median[2] - saved_pixels[x1 + r0, y1 + r0][2])) // 3
                if dist >= 1:
                    draw_result.point((x1 + r0, y1 + r0), black)
                elif (dist < 0.98):
                    draw_result.point((x1 + r0, y1 + r0), (color, color, color))
                else:
                    color = int(color * (50.0 - 50 * dist))
                    draw_result.point((x1 + r0, y1 + r0), (color, color, color))



    circled_image.save(new_log_picture())

    #resize to 1024*1024
    circled_image = circled_image.resize(req_size)
    circled_image.save(new_log_picture())

    #making deviant-colored points black, other - white
    binary_image = Image.new("RGB", req_size)
    output_pixels = circled_image.load()

    draw_result = ImageDraw.Draw(binary_image)

    brightnesses = []
    for x1 in range(req_width):
        for y1 in range(req_length):
            brightnesses.append(output_pixels[x1, y1][0])

    brightnesses.sort()

    k = 8000
    color_divider = brightnesses[len(brightnesses) - k]
    for x1 in range(req_width):
        for y1 in range(req_length):
            if output_pixels[x1, y1][0] >= color_divider:
                draw_result.point((x1, y1), white)
            else:
                draw_result.point((x1, y1), black)

    binary_image.save(new_log_picture())
    binary_image.save(tag_picture("last"))

def run_all():
    input_files = os.listdir(input_folder)
    for input_file in input_files:
        process_file(input_file)

if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    run_all()