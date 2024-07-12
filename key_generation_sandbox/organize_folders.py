#!/usr/bin/env python3
import os.path

by_folders = '../input/by_folders'
to_folder= '../input'
for d in os.listdir(by_folders):
	for f in os.listdir(by_folders + '/' + d):
		file_from = by_folders + '/' + d + '/' + f
		file_to = to_folder + '/' + d.zfill(4) + '_' + f
		print(file_from, file_to)
		os.rename(file_from, file_to)
