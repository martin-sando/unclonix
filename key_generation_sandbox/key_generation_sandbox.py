#!/usr/bin/env python
import sys
import os.path

input_folder = '../input'
output_folder = '../output'

def process_file(input_file):
	filename = input_file.split('.')[0]
	print(filename)

def run_all():
	input_files = os.listdir(input_folder)
	for input_file in input_files:
		process_file(input_file)

if __name__ == '__main__':
	os.makedirs(output_folder, exist_ok=True)
	run_all()
