import sys
import os.path
import bloblist_operations
import image_processing
import utils

def run_all(prefix, mask, reverse):
    input_files = sorted(os.listdir(utils.input_folder))
    hash = ""
    if reverse:
        input_files = input_files[::-1]
    for input_file in input_files:
        if '~' in input_file or mask not in input_file or not input_file.startswith(prefix):
            continue
        filename = input_file.split('.')[0]
        if not os.path.isfile(os.path.join(utils.bloblist_folder, filename + '.txt')):
            image_processing.process_photo(input_file, False)
        else:
            print("Reusing computed bloblist")
        hash = bloblist_operations.process_photo(input_file, True)
    return hash

def handle_image(name):
    utils.set_total_time()
    try:
        os.makedirs(utils.input_folder, exist_ok=True)
        os.makedirs(utils.output_folder, exist_ok=True)
        os.makedirs(utils.bloblist_folder, exist_ok=True)
        os.makedirs(utils.report_folder, exist_ok=True)
        os.makedirs(utils.time_folder, exist_ok=True)
        hash = run_all('', name, False)
    except:
        return "Error"
    return hash

if __name__ == '__main__':
    os.makedirs(utils.output_folder, exist_ok=True)
    os.makedirs(utils.bloblist_folder, exist_ok=True)
    os.makedirs(utils.report_folder, exist_ok=True)
    os.makedirs(utils.time_folder, exist_ok=True)
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
