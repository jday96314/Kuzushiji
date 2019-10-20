import re

with open('GroundTruthBoundingBoxes.csv') as bounding_boxes_file:
	lines = bounding_boxes_file.read().split('\n')

with open('PerfectObjectDetectorPredictions.csv', 'w') as submission_file:
	for line_index, line in enumerate(lines):
		if len(line) == 0:
			break

		if line_index == 0:
			submission_file.write(line + '\n')
		else:
			image_id, raw_bboxes = line.split(',')
			bounding_box_parsing_regex = r'(U\+[A-Z0-9]*) ([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)'
			ground_truth_bounding_boxes = re.findall(bounding_box_parsing_regex, raw_bboxes)
			detections = []
			for unicode_val, top_left_x_str, top_left_y_str, w_str, h_str in ground_truth_bounding_boxes:
				x_center = round(int(top_left_x_str) + int(w_str)/2)
				y_center = round(int(top_left_y_str) + int(h_str)/2)
				detections.append('{} {} {}'.format(
					unicode_val, x_center, y_center))
			submission_file.write(image_id + "," + " ".join(detections) + "\n")