# Attempt to process videos
# It is heavily unfinished and may contain bugs/weird behaviour
# It parses video into some (5, may be changed) frames and calculates hash from them
# Also it should contain flicker check, but right now it works poorly
import os
import sys
from PIL import Image, ImageDraw
import cv2
import image_processing, bloblist_operations
from Blob import Blob
import utils
import numpy as np
from math import sqrt, pi, cos, sin, exp, atan2

check_inside = utils.check_inside
black, blue, red, green, white = utils.black, utils.blue, utils.red, utils.green, utils.white
req_size, req_width, req_height, r = utils.req_size, utils.req_width, utils.req_height, utils.r
to_array, to_image = utils.to_array, utils.to_image
save = utils.save
run_experiment = utils.run_experiment


video_folder = utils.input_folder + '/video'
output_video_folder = utils.output_folder + '/video'

def process_video(input_file):
    filename = input_file.split('.')[0]
    utils.set_file_name(filename)
    utils.set_picture_number(0)
    utils.set_phase_time(1)
    utils.set_save_subfolder('')
    print('Processing ' + filename)

    vid = cv2.VideoCapture(os.path.join(video_folder, input_file))
    amount_of_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(5):
        vid.set(cv2.CAP_PROP_POS_FRAMES, (amount_of_frames * i) / 5 - 1)
        res, frame = vid.read()
        if res:
            img = Image.fromarray(frame, 'RGB')
            utils.save_no_number(img, 'frame' + str(i), filename)


def get_flicker_array(image, bloblist):
    image_pixels = image.load()
    res_array = np.zeros(128)
    for Blob in bloblist:
        r = int(max(sqrt((Blob.coords[0] - 512) ** 2 + (Blob.coords[1] - 512) ** 2) // 4, 2))
        brightness_sum = 0
        for x in range(-4, 5):
            for y in range(-4, 5):
                brightness_sum += image_pixels[Blob.coords[0] + x, Blob.coords[1] + y][0] + image_pixels[Blob.coords[0] + x, Blob.coords[1] + y][1] + image_pixels[Blob.coords[0] + x, Blob.coords[1] + y][2]
        req_brightness = 81 * 128 * 3
        if brightness_sum > req_brightness and r < 110:
            res_array[r-2] += 1
            res_array[r-1] += 4
            res_array[r] += 6
            res_array[r+1] += 4
            res_array[r+2] += 1
    return res_array

def check_flicker(flicker_arrays):
    total_array = np.zeros(128)
    for i in range(0, 128):
        for j in range(0, len(flicker_arrays)):
            total_array[i] += (flicker_arrays[j][i] / 5)

    tot_diff = 0
    for j in range(0, len(flicker_arrays)):
        for i in range(0, 128):
            tot_diff += abs(total_array[i] - flicker_arrays[j][i])
    if (tot_diff > 15):
        print("Video is valid")
        return True
    else:
        print("Video is fake")
        return False

def run_all(prefix, mask, reverse):
    input_videos = sorted(os.listdir(video_folder))
    if reverse:
        input_videos = input_videos[::-1]
    for input_file in input_videos:
        videoname = input_file.split('.')[0]
        os.makedirs(utils.output_folder + "/" + videoname, exist_ok=True)
        process_video(input_file)
        input_files = sorted(os.listdir(utils.output_folder + "/" + videoname))
        flicker_arrays = []
        hashes = []
        for input_file in input_files:
            if '~' in input_file or mask not in input_file or not input_file.startswith(prefix):
                continue
            filename = input_file.split('.')[0]
            if filename.startswith("p"):
                continue
            if not os.path.isfile(os.path.join(utils.bloblist_folder, filename + '.txt')):
                image_processing.process_photo(input_file, False, utils.output_folder + "/" + videoname)

            else:
                print("Reusing computed bloblist")
            bloblist = utils.get_blob_list(os.path.join(utils.bloblist_folder, filename + '.txt'))
            image = Image.open(os.path.join(utils.output_folder, filename + "_p03finding_circle_ph2.png"))
            flicker_arrays.append(get_flicker_array(image, bloblist))

            hash = bloblist_operations.process_photo(input_file, True)
            hashes.append(hash)
        check_flicker(flicker_arrays)




if __name__ == '__main__':
    os.makedirs(utils.output_folder, exist_ok=True)
    os.makedirs(utils.bloblist_folder, exist_ok=True)
    os.makedirs(utils.report_folder, exist_ok=True)
    os.makedirs(utils.time_folder, exist_ok=True)
    os.makedirs(output_video_folder, exist_ok=True)
    if os.path.exists(utils.hashes_file):
        os.remove(utils.hashes_file)
    utils.set_total_time()
    prefix = ''
    mask = ''
    reverse = False
    for arg in sys.argv:
        if arg.startswith('--prefix='):
            prefix = arg[9:]
        if arg.startswith('--mask='):
            mask = arg[7:]
        if arg.startswith('--reverse'):
            reverse = True
    run_all(prefix, mask, reverse)