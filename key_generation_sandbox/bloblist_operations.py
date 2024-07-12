import os.path

from PIL import Image, ImageDraw

bloblist_folder = '../bloblist'
output_folder = '../output-bloblist'

def test(filename):

    log_picture_number = 0
    def new_log_picture():
        nonlocal log_picture_number
        log_picture_number += 1
        return os.path.join(output_folder, filename + "#" + str(log_picture_number) + ".png")


    blue = (0, 0, 255)
    text_file = open(os.path.join(bloblist_folder, filename), 'r')
    image = Image.new("RGB", [1024, 1024])
    draw_result = ImageDraw.Draw(image)
    for text_line in text_file:
        numbers = text_line.split()
        x = int(float(numbers[0]))
        y = int(float(numbers[1]))
        sigma = int(float(numbers[2]))
        for i in range(sigma):
            draw_result.point((x, y+i), blue)
            draw_result.point((x, y-i), blue)
            draw_result.point((x+i, y), blue)
            draw_result.point((x-i, y), blue)
    image.save(new_log_picture())









if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    input_files = os.listdir(bloblist_folder)
    for input_file in input_files:
        test(input_file)