import os
import datetime
import shutil
from PIL import Image
import re

SCALED_IMAGE_WIDTHS = 512
SCALED_IMAGE_HEIGHTS = 512
USE_COLOR = False

def ConvertImageToResizedGreyscale(input_image):
	resized_image = input_image.resize((SCALED_IMAGE_WIDTHS, SCALED_IMAGE_HEIGHTS), Image.LANCZOS)
	if not USE_COLOR:
		return resized_image
			
	greyscale_resized_image = resized_image.convert('L')
	return greyscale_resized_image

# This could easily be parallelized for better performance.
def ConvertDirectoryContents(input_dir, output_dir):
	try:
		os.mkdir(output_dir)
		print('Created', output_dir)
	except FileExistsError:
		shutil.rmtree(output_dir)
		os.mkdir(output_dir)
		print('Cleared contents of', output_dir)

	print('Preprocessing images in', input_dir, '...')
	image_names = os.listdir(input_dir)
	for image_index, image_name in enumerate(image_names):
		if image_index%100 == 0:
			# Display status information.
			print('{}: Processing image {}/{}'.format(
				datetime.datetime.now(),
				image_index + 1,
				len(image_names)))

		# Read the image.
		original_image_path = os.path.join(input_dir, image_name)
		original_image = Image.open(original_image_path)
		
		# Convert the image.
		greyscale_resized_image = ConvertImageToResizedGreyscale(original_image)
		
		# Save the converted image.
		# jpg images contain compression artifacts, so pngs are used instead.
		output_image_path = os.path.join(output_dir, image_name).replace('jpg', 'png')
		greyscale_resized_image.save(output_image_path, 'PNG')

def RescaleGroundTruthCsv(input_csv_path, original_images_directory, output_csv_path):
	# CREATE AN OUTPUT FILE WITH THE CORRECT HEADER.
	output_csv = open(output_csv_path, 'w')
	output_csv.write('image_id,labels\n')

	# RESCALE ALL THE BOUNDING BOXES.
	# The first line is a header that can be ignored.
	for line in open(input_csv_path, 'r').readlines()[1:]:
		image_id, raw_bboxes = line.split(',')

		output_csv.write(image_id + ',')

		original_image_path = os.path.join(original_images_directory, image_id + '.jpg')
		original_image_width,original_image_height = Image.open(original_image_path).size

		bbox_parsing_regex = r'(U\+[A-Z0-9]*) ([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)'
		bboxes = re.findall(bbox_parsing_regex, raw_bboxes)
		for unicode_val, x, y, w, h in bboxes:
			width_scale_coef = SCALED_IMAGE_WIDTHS/original_image_width
			height_scale_coef = SCALED_IMAGE_HEIGHTS/original_image_height
			scaled_x, scaled_y = round(float(x)*width_scale_coef), round(float(y)*height_scale_coef)
			scaled_w, scaled_h = round(float(w)*width_scale_coef), round(float(h)*height_scale_coef)

			output_csv.write('{} {} {} {} {} '.format(
				unicode_val,scaled_x, scaled_y, scaled_w, scaled_h))

		output_csv.write('\n')

if __name__ == '__main__':
	# ConvertDirectoryContents(
	# 	input_dir = 'Datasets/train_images',
	# 	output_dir = 'Datasets/preprocessed_train_images')
	# ConvertDirectoryContents(
	# 	input_dir = 'Datasets/test_images',
	# 	output_dir = 'Datasets/preprocessed_test_images')
	# RescaleGroundTruthCsv(
	# 	input_csv_path = 'Datasets/train.csv',
	# 	original_images_directory = 'Datasets/train_images',
	# 	output_csv_path = 'Datasets/scaled_train.csv')

	ConvertDirectoryContents(
		input_dir = 'Datasets/train_images',
		output_dir = 'Datasets/Color/preprocessed_train_images')
	ConvertDirectoryContents(
		input_dir = 'Datasets/test_images',
		output_dir = 'Datasets/Color/preprocessed_test_images')
	