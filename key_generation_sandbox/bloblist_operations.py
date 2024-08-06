import os.path
import sys

import cv2
import imagehash
import json
import random
from key_generation_sandbox import (Blob, check_inside, to_image, to_array, input_folder, output_folder, bloblist_folder, report_folder, run_all,
                                    req_size, req_width, req_height, white, black, red, green, blue, hru_array)
from PIL import Image, ImageDraw
import numpy as np
from math import sqrt
from math import sqrt, pi, cos, sin, exp, atan2
from findiff import Gradient, Divergence, Laplacian, Curl
def get_blob_list(filename):
    text_read = open(filename)
    blobs=[]
    for line in text_read:
        blob = Blob.unpack(line)
        blobs.append(blob)
    return blobs

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
                color = red
            if i == 1:
                color = green
            if i == 2:
                color = blue
            draw.point(coords, color)
            for i in range(6):
                draw.point((coords[0] - i, coords[1]), color)
                draw.point((coords[0], coords[1] - i), color)
                draw.point((coords[0] + i, coords[1]), color)
                draw.point((coords[0], coords[1] + i), color)
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
def process_file(input_file, full_research_mode, mask):
    if mask == '':
        mask = 'unlabeled'
    filename = input_file.split('.')[0]
    print('Processing (phase 2) ' + filename)
    blobs_obj = get_blob_list(os.path.join(report_folder, mask, filename + '.txt'))
    new_circled_image = Image.open(os.path.join(report_folder, mask, filename + "_" + "brightened" + ".png"))
    circled_pixels = new_circled_image.load()

    log_picture_number = 0
    def save(image, tag):
        nonlocal log_picture_number
        log_picture_number += 1
        tag = 'p' + str(log_picture_number).zfill(2) + tag
        image.save(os.path.join(report_folder, mask, filename + "_" + tag + ".png"))
        image.save(os.path.join(report_folder, mask, tag + "_" + filename + ".png"))

    red_blobs = get_best_color(blobs_obj, 50, 0)
    green_blobs = get_best_color(blobs_obj, 50, 1)
    blue_blobs = get_best_color(blobs_obj, 50, 2)
    colors_blobs = [red_blobs, green_blobs, blue_blobs]
    triangles = find_triangles(req_width // 2, colors_blobs, (60, 60, 60), 2)
    copy_image = new_circled_image.copy()
    triangles_image = draw_triangles(copy_image, triangles, colors_blobs)
    save(triangles_image, 'chaos')


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

    if not full_research_mode:
        return


    log_picture = new_circled_image.copy()

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


if __name__ == '__main__':
    mask = ''
    for arg in sys.argv:
        if arg.startswith('--mask='):
            mask = arg[7:]
    run_all(mask, 2)
