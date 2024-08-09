import sys
import os.path
import bloblist_operations
import image_processing
import utils

input_folder, output_folder, bloblist_folder, report_folder = utils.input_folder, utils.output_folder, utils.bloblist_folder, utils.report_folder

def run_all(mask):
    input_files = os.listdir(input_folder)
    for input_file in sorted(input_files)[::-1]:
        if '~' in input_file or mask not in input_file:
            continue
        filename = input_file.split('.')[0]
        if not os.path.isfile(os.path.join(bloblist_folder, filename + '.txt')):
            image_processing.process_photo(input_file, False, mask)
        bloblist_operations.process_photo(input_file, False, mask)


if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bloblist_folder, exist_ok=True)
    os.makedirs(report_folder, exist_ok=True)
    mask = ''
    for arg in sys.argv:
        if arg.startswith('--mask='):
            mask = arg[7:]
    run_all(mask)
