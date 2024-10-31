import os.path
import cv2
import imagehash
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from math import sqrt, pi, cos, sin, exp, atan2
import utils
from image_processing import distinctiveness_coef, calculate_properties

check_inside = utils.check_inside
black, blue, red, green, white = utils.black, utils.blue, utils.red, utils.green, utils.white
req_size, req_width, req_height = utils.req_size, utils.req_width, utils.req_height
to_array, to_image = utils.to_array, utils.to_image
hru_array = utils.hru_array
save, save_report = utils.save, utils.save_report
run_experiment = utils.run_experiment
r = utils.r


def research_picture(image, blobs):
    image = image.copy()
    colors = {}
    for blob in blobs:
        colors[blob] = round(max(blob.bmp_128_7[0])), round(max(blob.bmp_128_7[3])), round(max(blob.bmp_128_7[6]))
    image = utils.draw_blobs(image, blobs, colors, mode_circle=True, only_distinctive=True)
    return image


def get_best_color(blobs, amount, color_num):
    color_dict = add_colors(blobs)
    colors = []
    for blob in blobs:
        if blob.dct_128_8 == None:
            continue
        colors.append(color_dict[blob][color_num])
    colors.sort()
    best_blobs = []
    for blob in blobs:
        if blob.dct_128_8 == None:
            continue
        if (color_dict[blob][color_num] >= colors[max(len(colors) - amount, 0)]):
            best_blobs.append(blob)
    return best_blobs

def add_colors(blobs):
    color_dict = dict()
    for blob in blobs:
        dct_array = blob.bmp_128_7
        if dct_array is None:
            continue
        colors = [0, 0, 0]
        sum = 0
        for i in range(7):
            for j in range(7):
                sum += hru_array[0][i * 7 + j] * dct_array[i][j]
        colors[0] = dct_array[0][0]
        sum = 0
        for i in range(7):
            for j in range(7):
                sum += hru_array[1][i * 8 + j] * dct_array[i][j]
        colors[1] = dct_array[6][0]
        sum = 0
        for i in range(7):
            for j in range(7):
                sum += hru_array[2][i * 8 + j] * dct_array[i][j]
        colors[2] = dct_array[0][6]
        color_dict[blob] = colors
    return color_dict

def color_picture(blobs):
    color_d = add_colors(blobs)
    result_image = Image.new("RGB", [128, 128])
    draw_result = ImageDraw.Draw(result_image)
    for blob in blobs:
        if blob.dct_128_8 is None:
            continue
        coord_1 = int(color_d[blob][0] / 30 + 64)
        coord_2 = int(color_d[blob][1] / 30 + 64)
        draw_result.point((coord_1, coord_2), white)
    return result_image


best_triangle = None
def find_triangles(r, best_blobs_color, req_angles, threshold):
    triangle = (0, 0, 0)
    best_score = 1000
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
                    cur_score += abs(dist_12 - 500) / 50 + abs(dist_23 - 500) / 50 + abs(dist_13 - 500) / 50
                    coord_centre = ((coord1[0] + coord2[0] + coord3[0]) / 3, (coord1[1] + coord2[1] + coord3[1]) / 3)
                    cur_score += sqrt(((coord_centre[0]) / 50) ** 2 + ((coord_centre[1]) / 50) ** 2)
                    if cur_score < best_score:
                        triangle= (blob1, blob2, blob3)
                        best_score = cur_score
    triangles = [triangle]
    global best_triangle
    best_triangle = triangle
    #print(best_score)
    return triangles

def draw_triangles(image, triangles, best_blobs_color):
    draw = ImageDraw.Draw(image)
    for (blob1, blob2, blob3) in triangles:
        draw.line([(blob1.coords[0], blob1.coords[1]), (blob2.coords[0], blob2.coords[1])], fill="red", width=0)
        draw.line([(blob2.coords[0], blob2.coords[1]), (blob3.coords[0], blob3.coords[1])], fill="red", width=0)
        draw.line([(blob3.coords[0], blob3.coords[1]), (blob1.coords[0], blob1.coords[1])], fill="red", width=0)

    for clr in range(3):
        for blob in best_blobs_color[clr]:
            coords = blob.coords
            color = (0, 0, 0)
            if clr == 0:
                color = red
            if clr == 1:
                color = green
            if clr == 2:
                color = blue
            draw.point(coords, color)
            for i in range(6):
                draw.point((coords[0] - i, coords[1]), color)
                draw.point((coords[0], coords[1] - i), color)
                draw.point((coords[0] + i, coords[1]), color)
                draw.point((coords[0], coords[1] + i), color)
    return image

def get_field_image(measure_field, input_image, precision, contrast=128):
    r = req_width // 2
    result_array = np.zeros((req_width, req_height, 3))
    dct_sum = np.zeros((128, 128))
    dct_sum[0, 0] = 1
    req_dots = []
    for x in range(req_width // precision):
        for y in range(req_height // precision):
            req_dots.append((x * precision, y * precision))

    total_brightness = 0

    for dot in req_dots:
        coord_0 = int(dot[0])
        coord_1 = int(dot[1])
        dist = sqrt((coord_0 - r) ** 2 + (coord_1 - r) ** 2) / r
        if dist > 0.8:
            continue
        result_array[coord_0, coord_1] = measure_field(input_image, (coord_0, coord_1))
        total_brightness += abs(result_array[coord_0, coord_1][0]) + abs(result_array[coord_0, coord_1][1]) + abs(result_array[coord_0, coord_1][2])

    scaling = total_brightness / (len(req_dots) * contrast)

    field_image = Image.new("RGB", [req_width, req_height])
    draw_result = ImageDraw.Draw(field_image)
    for x in range(req_width // precision):
        for y in range(req_height // precision):
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
                            x1 / precision) * (y1 / precision)
                    result_array[coord_0, coord_1] = comp_1 + comp_2 + comp_3 + comp_4
    for x in range(req_width):
        for y in range(req_height):
            color = (int(128 + result_array[x, y][0] / scaling), int(128 + result_array[x, y][1] / scaling), int(128 + result_array[x, y][2] / scaling))
            draw_result.point((x, y), color)

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


def find_draw_triangles(image, blobs_obj):
    add_colors(blobs_obj)

    red_blobs = get_best_color(blobs_obj, 5, 0)
    green_blobs = get_best_color(blobs_obj, 5, 1)
    blue_blobs = get_best_color(blobs_obj, 5, 2)
    colors_blobs = [red_blobs, green_blobs, blue_blobs]
    triangles = find_triangles(req_width // 2, colors_blobs, (60, 60, 60), 50)
    copy_image = image.copy()
    triangles_image = draw_triangles(copy_image, triangles, colors_blobs)
    return triangles_image



def generate_some_fields(image, blobs_obj):
    def example_measure(image, coords):
        square = utils.get_rotated_surroundings(image, coords)
        result = calculate_properties(square)
        b = result.bmp_128_7
        d = result.dct_128_8
        x = max([abs(x) for x in sum(d, [])])
        color = (np.sign(x), 0, 0)
        return color

    field_image = get_field_image(example_measure, image, precision=10, contrast=70)
    return field_image

def get_dct(image, dct_size):
    pixels = image.load()
    array_image = np.zeros((req_width, req_height))
    for x1 in range(req_width):
        for y1 in range(req_height):
            array_image[x1, y1] = pixels[x1, y1][0] + 0.0

    dct_array = cv2.dct(array_image)

    dct_image = Image.new("RGB", [dct_size, dct_size])
    draw_result = ImageDraw.Draw(dct_image)
    for x1 in range(dct_size):
        for y1 in range(dct_size):
            color = int(dct_array[x1, y1])
            draw_result.point((x1, y1), (color, color, color))

    return dct_image

def draw_distinctiveness(image, blobs_obj):
    image = image.copy()
    pixels = to_array(image)
    blobs_dict = {}
    draw_result = ImageDraw.Draw(image)
    for blob in blobs_obj:
        xc = round(blob.coords[0])
        yc = round(blob.coords[1])
        length = blob.size * distinctiveness_coef()
        length2 = blob.size * 0.8
        cnt = 0
        brightness = 0
        brightness2 = 0
        for i in range(-int(length), int(length) + 1):
            for j in range(-int(length), int(length) + 1):
                dist = sqrt((i) ** 2 + (j) ** 2)
                if (length - 1) <= dist <= (length):
                    cnt += 1
                    brightness += max(pixels[xc + i, yc + j] - 20, 0)
                    draw_result.point((xc + i, yc + j), green)
                # if (length2 - 1) <= dist <= (length2):
                #     cnt += 1
                #     brightness2 += max((170 - pixels[xc + i, yc + j]), 0)
                #     draw_result.point((xc + i, yc + j), green)
        brightness *= (brightness2 + 300)
        brightness /= 10000
        blobs_dict[blob] = (int(brightness), 0, 256 - int(brightness))
    image = utils.draw_blobs(image, blobs_obj, blobs_dict, mode_circle=True)
    return image


def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 4.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(image, filters):
    # This general function is designed to apply filters to our image
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(image)
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image

    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(utils.to_array_3d(image), depth, kern)  #Apply filter to image

        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        for x in range(req_width):
            for y in range(req_height):
                for z in range(3):
                    newimage[x, y, z] = max(image_filter[x, y, z], newimage[x, y, z])
        save(utils.to_image_3d(image_filter, req_width, req_height), 'gabor')
    return utils.to_image_3d(newimage, req_width, req_height)
def apply_gabor(image):
    filters = create_gaborfilter()
    image = apply_filter(image, filters)
    return image

def rotated(image) :
    global best_triangle
    angle = atan2((best_triangle[0].coords[1] - r), (best_triangle[0].coords[0] - r))
    rot_image = image.rotate((angle * (180 / pi)))
    return rot_image

def affine_transform(image, src, dst):
    image_array = to_array(image)
    rows, cols = image_array.shape[:2]
    transform_matrix = cv2.getAffineTransform(src, dst)
    transformed_array = cv2.warpAffine(image_array, transform_matrix, (cols,rows))
    transformed_image = to_image(transformed_array)
    return transformed_image


def dct_hash(image, coords=(0, 0, req_width, req_height), hash_size=8, highfreq_factor=4):
    image = image.crop((coords[0], coords[1], coords[2], coords[3]))
    if hash_size < 2:
        raise ValueError('Hash size must be greater than or equal to 2')

    import scipy.fftpack
    img_size = hash_size * highfreq_factor
    image = image.convert('L').resize((img_size, img_size))
    pixels = np.asarray(image)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    hash = ""
    for i in range(hash_size):
        for j in range(hash_size):
            if (dctlowfreq[i][j] > 0):
                hash += '1'
            else:
                hash += '0'
    return hash



def process_photo(input_file, full_research_mode):
    filename = input_file.split('.')[0]
    utils.set_file_name(filename)
    utils.set_phase_time(2)
    utils.set_picture_number(utils.image_processing_picture_number_end)
    utils.set_save_subfolder('report')
    print('Processing (phase 2) ' + filename)
    blobs_obj = utils.get_blob_list(os.path.join(utils.bloblist_folder, filename + '.txt'))
    image = Image.open(utils.get_result_name())

    run_experiment(research_picture, image, blobs_obj)

    run_experiment(draw_distinctiveness, image, blobs_obj)

    #run_experiment(color_picture, blobs_obj)

    # run_experiment(generate_some_fields, image, blobs_obj)
    run_experiment(find_draw_triangles, image, blobs_obj)

    src = np.float32([[best_triangle[0].coords[1], best_triangle[0].coords[0]],
                      [best_triangle[1].coords[1], best_triangle[1].coords[0]],
                      [best_triangle[2].coords[1], best_triangle[2].coords[0]]])
    # triangle_coords = [[712, 512], [412, 339], [412, 685]]
    # margin = 400
    triangle_coords = [[1000, 512], [24, 24], [24, 1000]]
    margin = 50
    gap = 50
    rectangle_coords = ((margin, margin), (req_width - margin, req_height - margin))
    font = ImageFont.truetype("arial.ttf", gap // 2)
    image = run_experiment(affine_transform, image, src, np.float32(triangle_coords))

    checkpoints = []
    for x in range(rectangle_coords[0][0] + gap, rectangle_coords[1][0] - gap, 2 * gap):
        for y in range(rectangle_coords[0][1] + gap, rectangle_coords[1][1] - gap, 2 * gap):
            checkpoints.append((x, y))

    def area_for_hashing(image):
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for t in triangle_coords:
            draw.circle((t[1], t[0]), gap // 2, outline=blue)
        for xy in checkpoints:
            h = utils.bin2hex(dct_hash(image, (xy[0], xy[1], xy[0] + 2*gap, xy[1] + 2*gap), 4))
            draw.text((xy[0] + gap // 2, xy[1] + gap // 2), h, font=font)
            draw.circle(xy, 3, outline=green)
        draw.rectangle(rectangle_coords, outline=blue)
        return image
    run_experiment(area_for_hashing, image)

    if not full_research_mode:
        return

    def draw_only_blobs():
        return utils.draw_blobs(Image.new("RGB", [req_width, req_height]), blobs_obj, mode_image=True)
    run_experiment(draw_only_blobs)

    #run_experiment(get_dct, image, 32)

    compressed_size = 64
    image = image.resize((compressed_size, compressed_size))
    save_report(image, "hash_compress")
    the_hash = utils.with_control(str(imagehash.phash(image)).rjust(16, '0'))
    the_hash += '_' + utils.with_control(utils.bin2hex(dct_hash(image, (0, 0, compressed_size, compressed_size))))
    print(the_hash)
    with open(utils.hashes_file, 'a') as f:
        print(filename + '\t' + the_hash, file=f)

    utils.set_last_time('finishing_labor')
    return the_hash