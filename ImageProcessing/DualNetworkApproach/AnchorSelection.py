from sklearn.cluster import KMeans
import re
import numpy as np

def ComputeAnchorSizes(csv_filepath):
	scaled_bounding_box_sizes = []
	for line in open(csv_filepath).readlines()[1:]:
		image_id, raw_bounding_box_data = line.split(',')
		bounding_box_parsing_regex = r'(U\+[A-Z0-9]*) ([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)'
		detected_bounding_boxes = re.findall(bounding_box_parsing_regex, raw_bounding_box_data)

		for unicode_val, x, y, w, h in detected_bounding_boxes:
			scaled_bounding_box_sizes.append([w, h])

	model = KMeans(n_clusters=9).fit(scaled_bounding_box_sizes)
	anchors = np.round(model.cluster_centers_)
	return anchors