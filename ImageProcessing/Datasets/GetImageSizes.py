import os
from PIL import Image

with open('TestingImageSizes.csv', 'w') as output_file:
	output_file.write('image_id,image_width,image_height\n')

	IMAGE_DIR = 'test_images'
	image_file_names = os.listdir(IMAGE_DIR)
	for image_file_name in image_file_names:
		image_id = image_file_name.replace('.jpg', '')
		image_width, image_height = Image.open(os.path.join(IMAGE_DIR, image_file_name)).size
		output_file.write('{}, {}, {}\n'.format(
			image_id,
			image_width,
			image_height))
