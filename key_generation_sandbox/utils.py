import os
import cv2
import random
from math import atan2, pi, sqrt, exp
import numpy as np
from PIL import Image, ImageDraw
from Blob import Blob
import datetime

input_folder = '../input'
output_folder = '../output'
bloblist_folder = output_folder + '/bloblist'
report_folder = output_folder + '/report'
time_folder = output_folder + '/time'
hashes_file = output_folder + '/hashes.txt'

def prepare():
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bloblist_folder, exist_ok=True)
    os.makedirs(report_folder, exist_ok=True)
    os.makedirs(time_folder, exist_ok=True)

prepare()

white = (255, 255, 255)
gray = (127, 127, 127)
black = (0, 0, 0)
blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
req_width = 1024
req_height = 1024
req_size = (req_width, req_height)
r = req_width // 2

image_processing_picture_number_result = 10
image_processing_picture_number_end = 20

random.seed(566)
hru_array = []
for i in range(20):
    hru = [random.gauss(mu=0.0, sigma=1.0) for _ in range(64)]
    hru_array.append(hru)

def get_blob_info(image, coords, blobs, mask_num, filename, hru = hru_array[0], bound_1 = 0, bound_2 = 7, cutter_size=128):
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

    draw_result = ImageDraw.Draw(image)
    draw_result.point(closest_blob.coords, blue)
    x = int(closest_blob.coords[0])
    y = int(closest_blob.coords[1])
    draw_result.point((x, y), (0, 0, 255))
    for i in range(int(closest_blob.size)):
        draw_result.point((x, y + i), blue)
        draw_result.point((x, y - i), blue)
        draw_result.point((x + i, y), blue)
        draw_result.point((x - i, y), blue)
    r = image.width // 2
    angle = atan2((y - r), (x - r))
    blob_img = image.crop(
        (x - cutter_size, y - cutter_size, x + cutter_size,
         y + cutter_size))
    rot_image = blob_img.rotate((angle * (180 / pi)))

    blob_img = rot_image.crop(
        (int(cutter_size / 2), int(cutter_size / 2), int(cutter_size * (3 / 2)), int(cutter_size * (3 / 2))))
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
    log("bmp_128_7 is ")
    log_matrix(closest_blob.bmp_128_7)

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
def to_array(input_image):
    array_image = np.zeros((input_image.width, input_image.height))
    image_pixels = input_image.load()

    for x1 in range(input_image.width):
        for y1 in range(input_image.height):
            array_image[x1, y1] = image_pixels[x1, y1][0]
    return array_image

def to_array_3d(input_image):
    array_image = np.zeros((input_image.width, input_image.height, 3))
    image_pixels = input_image.load()

    for x1 in range(input_image.width):
        for y1 in range(input_image.height):
            array_image[x1, y1] = image_pixels[x1, y1]
    return array_image

def to_image(input_array, width=req_width, length=req_height):
    image = Image.new("RGB", [width, length])
    draw_result = ImageDraw.Draw(image)
    for x1 in range(width):
        for y1 in range(length):
            color = int(input_array[x1, y1])
            if input_array[x1, y1] is tuple:
                color = int((input_array[x1, y1][0] + input_array[x1, y1][1] + input_array[x1, y1][2]) / 3)
            draw_result.point((x1, y1), (color, color, color))
    return image

def to_image_3d(input_array, width, length):
    image = Image.new("RGB", [width, length])
    draw_result = ImageDraw.Draw(image)
    for x1 in range(width):
        for y1 in range(length):
            color = (int(input_array[x1, y1][0]), int(input_array[x1, y1][1]), int(input_array[x1, y1][2]))
            draw_result.point((x1, y1), color)
    return image


def transpose(image):
    result_image = Image.new("RGB", [image.height, image.width])
    draw_result = ImageDraw.Draw(result_image)
    image_pixels = image.load()
    for x1 in range(image.width):
        for y1 in range(image.height):
            color = int(image_pixels[x1, y1][0])
            draw_result.point((y1, x1), (color, color, color))
    return result_image
filename = ''

def set_file_name(name):
    global filename
    filename = name

log_picture_number = 0

def set_picture_number(number):
    global log_picture_number
    log_picture_number = number
def save(image, tag, subfolder=''):
    global log_picture_number
    global filename
    tag = 'p' + str(log_picture_number).zfill(2) + tag
    if subfolder == '':
        image.save(os.path.join(output_folder, filename + "_" + tag + ".png"))
        image.save(os.path.join(output_folder, tag + "_" + filename + ".png"))
    else:
        image.save(os.path.join(output_folder, subfolder, filename + "_" + tag + ".png"))
        image.save(os.path.join(output_folder, subfolder, tag + "_" + filename + ".png"))
    log_picture_number += 1

def save_no_number(image, tag, subfolder=''):
    global filename
    if subfolder == '':
        image.save(os.path.join(output_folder, filename + "_" + tag + ".png"))
    else:
        image.save(os.path.join(output_folder, subfolder, filename + "_" + tag + ".png"))

def save_report(image, tag):
    save(image, tag, "report")


result_tag = 'processed'

def get_result_name():
    return os.path.join(output_folder, filename + "_" + "p" + str(image_processing_picture_number_result).zfill(2) + result_tag + ".png")

start_time = None
phase_time = None
last_time = None
time_file = None

def set_total_time():
    global start_time
    time = datetime.datetime.now()
    start_time = time
def set_phase_time(phase_num):
    global time_file
    time_file = open(os.path.join(time_folder, filename + '.txt'), 'a')
    global phase_time
    global last_time
    time = datetime.datetime.now()
    last_time = time
    phase_time = time
    time_file.write("Starting phase " + str(phase_num) + ", now is " + str(time) + "\n")
def set_last_time(log):
    global time_file
    global start_time
    global last_time
    time = datetime.datetime.now()
    time_file.write("Finished doing " + log + "\n")
    time_file.write(", now is " + str(time) + ", " + str(time - last_time) + " elapsed since last measure, " + '\n')
    time_file.write(str(time - phase_time) + " elapsed total since phase start, " + str(time - start_time) + " elapsed total since work start" + '\n')
    last_time = time

save_subfolder = ''
def set_save_subfolder(subfolder):
    global save_subfolder
    save_subfolder = subfolder

def run_experiment(method, *method_args):
    method_name = method.__name__
    result_image = method(*method_args)
    save(result_image, method_name, save_subfolder)
    set_last_time(method_name)
    return result_image

def draw_blobs(image, blobs_list, blobs_dict=None, mode_plus=False, mode_image=False, mode_circle=False, mode_circumference=False, only_distinctive=False):
    draw_result = ImageDraw.Draw(image)
    for blob in blobs_list:
        if only_distinctive and blob.distinctiveness == 0:
            continue
        x = int(blob.coords[0])
        y = int(blob.coords[1])
        sigma = blob.size
        brightness = blob.brightness
        color = blue
        if blobs_dict is not None:
            color = blobs_dict[blob]

        if mode_plus:
            draw_result.point((x, y), color)
            for i in range(int(blob.size)):
                draw_result.point((x, y + i), color)
                draw_result.point((x, y - i), color)
                draw_result.point((x + i, y), color)
                draw_result.point((x - i, y), color)
        if mode_image:
            for i in range(-int(sigma), int(sigma) + 1):
                for j in range(-int(sigma), int(sigma) + 1):
                    if check_inside(x + i, y + j, req_height, req_width):
                        dist = sqrt((i) ** 2 + (j) ** 2)
                        if dist <= sigma:
                            color = int(exp(-((dist / sigma) / 2)) * brightness)
                            #draw_result.point((x + i, y + j), (color, color, color))
                            draw_result.point((x + i, y + j), white)
        if mode_circle:
            for i in range(-int(sigma), int(sigma) + 1):
                for j in range(-int(sigma), int(sigma) + 1):
                    if check_inside(x + i, y + j, req_height, req_width):
                        dist = sqrt((i) ** 2 + (j) ** 2)
                        if dist <= sigma:
                            draw_result.point((x + i, y + j), color)
        if mode_circumference:
            for i in range(-int(sigma), int(sigma) + 1):
                for j in range(-int(sigma), int(sigma) + 1):
                    if check_inside(x + i, y + j, req_height, req_width):
                        dist = sqrt((i) ** 2 + (j) ** 2)
                        if (sigma - 1) <= dist <= sigma:
                            draw_result.point((x + i, y + j), color)
    return image

def get_rotated_surroundings(image, coords, cutter_size=128):
    coord_0 = coords[0]
    coord_1 = coords[1]
    angle = atan2((coord_1 - r), (coord_0 - r))
    blob_img = image.crop(
        (coord_0 - cutter_size, coord_1 - cutter_size, coord_0 + cutter_size, coord_1 + cutter_size))
    rot_image = blob_img.rotate((angle * (180 / pi)))
    surr_img = rot_image.crop(
        (int(cutter_size / 2), int(cutter_size / 2), int(cutter_size * (3 / 2)), int(cutter_size * (3 / 2))))
    return surr_img

def bin2hex(str_bin):
    return hex(int(str_bin, 2))[2:].rjust((len(str_bin) + 3) // 4, '0')

def with_control(str_hex):
    return str_hex + '[' + str(int(str_hex, 16) * 566 % 9566239) + ']'
