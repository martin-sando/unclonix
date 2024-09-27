#!/usr/bin/env python3
import os.path
import utils

by_folders = utils.input_folder + '/by_folders'
to_folder= utils.input_folder
for d in os.listdir(by_folders):
	for f in os.listdir(by_folders + '/' + d):
		file_from = by_folders + '/' + d + '/' + f
		file_to = to_folder + '/' + d.zfill(4) + '_' + f
		print(file_from, file_to)
		os.rename(file_from, file_to)
