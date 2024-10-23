import sys
import os.path
import bloblist_operations
import image_processing
import utils

def run(input_file):
    filename = input_file.split('.')[0]
    if os.path.isdir(os.path.join(utils.input_folder, filename)):
        return ""
    if not os.path.isfile(os.path.join(utils.bloblist_folder, filename + '.txt')):
        image_processing.process_photo(input_file, False)
    else:
        print("Reusing computed bloblist")
    return bloblist_operations.process_photo(input_file, True)


def run_all(prefix, mask, reverse):
    if os.path.exists(utils.hashes_file):
        os.remove(utils.hashes_file)
    utils.set_total_time()
    input_files = sorted(os.listdir(utils.input_folder))
    if reverse:
        input_files = input_files[::-1]
    for input_file in input_files:
        if '~' in input_file or mask not in input_file or not input_file.startswith(prefix):
            continue
        run(input_file)


if __name__ == '__main__':
    prefix = ''
    mask = ''
    reverse = False
    for arg in sys.argv:
        if arg == 'bot':
            import bot
            bot.run()
            exit()
        if arg.startswith('--prefix='):
            prefix = arg[9:]
        if arg.startswith('--mask='):
            mask = arg[7:]
        if arg.startswith('--reverse'):
            reverse = True
    run_all(prefix, mask, reverse)
