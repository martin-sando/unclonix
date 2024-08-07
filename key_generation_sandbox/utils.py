import os
import cv2
import random
from math import atan2, pi
import numpy as np
from PIL import Image, ImageDraw
from Blob import Blob

input_folder = '../input'
output_folder = '../output'
bloblist_folder = output_folder + '/bloblist'
report_folder = output_folder + '/report'

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
def to_image(input_array, width, length):
    image = Image.new("RGB", [width, length])
    draw_result = ImageDraw.Draw(image)
    for x1 in range(width):
        for y1 in range(length):
            color = int(input_array[x1, y1])
            draw_result.point((x1, y1), (color, color, color))
    return image

filename = ''

def set_file_name(name):
    global filename
    filename = name

log_picture_number = 0

def set_picture_number(number):
    global log_picture_number
    log_picture_number = number
def save(image, tag):
    global log_picture_number
    global filename
    tag = 'p' + str(log_picture_number).zfill(2) + tag
    image.save(os.path.join(output_folder, filename + "_" + tag + ".png"))
    image.save(os.path.join(output_folder, tag + "_" + filename + ".png"))
    log_picture_number += 1

result_tag = '_processed'

def get_result_name():
    return os.path.join(output_folder, filename + "_" + "p" + str(image_processing_picture_number_result).zfill(2) + "_processed" + ".png")