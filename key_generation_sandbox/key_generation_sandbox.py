import sys
import os.path
import bloblist_operations
import image_processing
import utils

input_folder, output_folder, bloblist_folder, report_folder, time_folder = utils.input_folder, utils.output_folder, utils.bloblist_folder, utils.report_folder, utils.time_folder

def run_all(prefix, mask, reverse):
    input_files = sorted(os.listdir(input_folder))
    if reverse:
        input_files = input_files[::-1]
    for input_file in input_files:
        if '~' in input_file or mask not in input_file or not input_file.startswith(prefix):
            continue
        filename = input_file.split('.')[0]
        if not os.path.isfile(os.path.join(bloblist_folder, filename + '.txt')):
            image_processing.process_photo(input_file, False)
        else:
            print("Reusing computed bloblist")
        bloblist_operations.process_photo(input_file, True)


if __name__ == '__main__':
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(bloblist_folder, exist_ok=True)
    os.makedirs(report_folder, exist_ok=True)
    os.makedirs(time_folder, exist_ok=True)
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
