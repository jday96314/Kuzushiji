from torch.utils.data import Dataset
import re
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import torch

# For char ords are reserved for the following:
# 0 = sos (start of string)
# 1 = eos (end of string)
# 2 = pad (padding)
# 3 = NA (Not a real char)
RESERVED_CHAR_COUNT = 4

def ObjectDetectorLabelsToParallelVectors(object_detections_str, sorted_charset, image_width, image_height):	
	location_vectors = []
	label_vectors = []
	if type(object_detections_str) != str:
		return location_vectors, label_vectors

	# # The two extra dimensions are for EOS and SOS.
	# label_dim_count = len(sorted_charset) + 2

	label_retrieval_regex = r'(U\+[A-Z0-9]*) ([0-9]*) ([0-9]*)'
	labels = re.findall(label_retrieval_regex, object_detections_str)
	for unicode_val, x, y in labels:
		# one_hot_char_type = np.zeros(label_dim_count)
		# one_hot_char_type[np.argwhere(sorted_charset == unicode_val)] = 1
		# label_vectors.append(one_hot_char_type)

		# The 3 is added because 0, 1, and 2 are reserved for sos, eos, and pad.
		char_type = np.array(np.argwhere(sorted_charset == unicode_val)[0] + RESERVED_CHAR_COUNT) 
		label_vectors.append(char_type)

		normalized_location = np.array(
		[
			float(x)/image_width,
			float(y)/image_height
		])
		location_vectors.append(normalized_location)

	return np.array(location_vectors), np.array(label_vectors)

def GetLabelsSortedByReadOrder(unordered_char_location_vectors, unordered_char_label_vectors):
	if len(unordered_char_location_vectors) < 2:
		return unordered_char_location_vectors, unordered_char_label_vectors

	# GROUP THE LABELS BY COLUMN.
	# Differences in the x direction should be far more damning than in the y.
	x_stretch_coef = 15
	unordered_char_location_vectors[:,0] *= x_stretch_coef
	column_detector = DBSCAN(eps = .15, min_samples = 1)
	char_column_labels = column_detector.fit_predict(unordered_char_location_vectors)
	unordered_char_location_vectors[:,0] /= x_stretch_coef

	# DETERMINE THE ORDER IN WHICH THE COLUMNS SHOULD BE READ.
	# They should typically be read from right to left.
	# Y coords are taken into consideration for robustness in the event a column is erroneiously
	# split in two and the image is tilted such that the lower half is to the right of the top
	# half.
	column_ids_and_centers = [] # [[id, center_x, center_y], ...]
	for column_id in range(max(char_column_labels) + 1):
		column_char_locations = unordered_char_location_vectors[
			np.argwhere(char_column_labels == column_id)]
		column_char_locations = np.reshape(column_char_locations, (len(column_char_locations), 2))
		average_location_of_char_in_col = np.mean(column_char_locations, axis = 0)
		column_ids_and_centers.append(np.concatenate([[column_id], average_location_of_char_in_col]))
	column_ids_and_centers_by_read_order = np.array(sorted(
		column_ids_and_centers,
		key = lambda vect: vect[2] - 4*vect[1]))
	column_ids_ordered_by_read_order = column_ids_and_centers_by_read_order[:,0]
	
	# CONCATENATE THE LOCATIONS AND THE LABELS.
	# This is done to simplify the location based sorting.
	unordered_concatenated_char_descriptions = np.concatenate(
		[unordered_char_location_vectors, unordered_char_label_vectors],
		axis = 1)

	# DETERMINE THE ORDER IN WHICH THE CHARACTERS SHOULD BE READ.
	# They should be read from top to bottom.
	image_locations_in_read_order = []
	image_labels_in_read_order = []
	for column_id in column_ids_ordered_by_read_order:
		# SORT THE CHAR DESCRIPTIONS.
		column_char_descriptions = unordered_concatenated_char_descriptions[
			np.argwhere(char_column_labels == column_id)]
		column_char_descriptions = np.reshape(
			column_char_descriptions, 
			(np.shape(column_char_descriptions)[0], np.shape(column_char_descriptions)[2]))
		column_char_descriptions_in_read_order = np.array(sorted(
			column_char_descriptions,
			key = lambda vect: vect[1]))

		# SPLIT OUT THE LOCATIONS AND LABELS.
		image_locations_in_read_order.append(column_char_descriptions_in_read_order[:,:2])
		image_labels_in_read_order.append(column_char_descriptions_in_read_order[:,2:])

	image_locations_in_read_order = np.concatenate(image_locations_in_read_order)
	image_labels_in_read_order = np.concatenate(image_labels_in_read_order)

	return image_locations_in_read_order, image_labels_in_read_order

class CharSequencesDataset(Dataset):
	def __init__(
			self, 
			perfect_detector_predictions_path, 
			image_sizes_path, 
			image_ids, 
			sorted_charset,
			data_mangler,
			max_sequence_length):
		perfect_detector_predictions = pd.read_csv(perfect_detector_predictions_path)
		perfect_detector_predictions = perfect_detector_predictions[perfect_detector_predictions['image_id'].isin(image_ids)].sort_values(by = 'image_id')
		self.GroundTruthSequences = perfect_detector_predictions['labels'].values
		self.ImageSizes = pd.read_csv(image_sizes_path).sort_values(by = 'image_id')
		self.ImageSizes = self.ImageSizes[self.ImageSizes['image_id'].isin(image_ids)]
		self.ImageSizes = list(zip(
			self.ImageSizes['image_width'].values, 
			self.ImageSizes['image_height'].values))
		self.SortedCharset = sorted_charset
		self.ImageIds = image_ids
		self.DataMangler = data_mangler
		self.MaxSequenceLength = max_sequence_length	

		self.SOS_char_label = [0]
		self.SOS_location = [1, 0] # Top right

		self.EOS_char_label = [1]
		self.EOS_location = [0, 1] # Bottom left

	def __len__(self):
		return len(self.ImageSizes)

	def __getitem__(self, image_index):
		# GET THE ORDERED SEQUENCE.
		image_width, image_height = self.ImageSizes[image_index]
		ground_truth_image_labels = self.GroundTruthSequences[image_index]
		unordered_character_locations, unordered_character_labels = ObjectDetectorLabelsToParallelVectors(
			ground_truth_image_labels, self.SortedCharset, image_width, image_height)
		character_locations, character_labels = GetLabelsSortedByReadOrder(
			unordered_character_locations, unordered_character_labels)

		# IF NECESSARY, TRIM THE SEQUENCE.
		if len(character_locations) > self.MaxSequenceLength:
			start_index = np.random.randint(0, len(character_labels) - self.MaxSequenceLength)
			end_index = start_index + self.MaxSequenceLength
			character_locations = character_locations[start_index:end_index]
			character_labels = character_labels[start_index:end_index]

		# CONVERT THE SEQUENCE TO TENSORS. 
		if len(character_locations) > 0:
			ground_truth_location_sequence = torch.tensor(
				np.concatenate([
					character_locations, 
					[self.EOS_location]]),
				dtype = torch.float32)
			ground_truth_label_sequence = torch.tensor(
				np.concatenate([
					character_labels, 
					[self.EOS_char_label]]),
				dtype = torch.long)
		else:
			ground_truth_location_sequence = torch.tensor([self.EOS_location], dtype = torch.float32)
			ground_truth_label_sequence = torch.tensor([self.EOS_char_label], dtype = torch.long)

		# MANGLE THE SEQUENCE.
		(
			modified_ground_truth_location_sequence, 
			modified_ground_truth_label_sequence, 
			skipped_char_locations,
			skipped_char_labels,
			mangled_location_sequence, 
			mangled_label_sequence,
			detection_confidences
		) = self.DataMangler.GetMangledSequence(
			ground_truth_location_sequence, 
			ground_truth_label_sequence,
			image_width,
			image_height)
		return (
			modified_ground_truth_location_sequence, 
			modified_ground_truth_label_sequence, 
			skipped_char_locations,
			skipped_char_labels,
			mangled_location_sequence, 
			mangled_label_sequence,
			detection_confidences)