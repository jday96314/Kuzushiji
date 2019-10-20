import torch
import math
import pandas as pd
import numpy as np
import pickle
from CharSequencesDataset import GetLabelsSortedByReadOrder
from Model import ClassificationCorrector
from tqdm import tqdm
import sys
import re

DEVICE = torch.device("cuda")
# For char ords are reserved for the following:
# 0 = sos (start of string)
# 1 = eos (end of string)
# 2 = pad (padding)
# 3 = NA (Not a real char)
RESERVED_CHAR_COUNT = 4

def ObjectDetectorLabelsWithConfidenceToParallelVectors(object_detections_str, sorted_charset, image_width, image_height):	
	location_vectors = []
	label_vectors = []
	confidences = []
	if type(object_detections_str) != str:
		return location_vectors, label_vectors, confidences

	# # The two extra dimensions are for EOS and SOS.
	# label_dim_count = len(sorted_charset) + 2

	label_retrieval_regex = r'(U\+[A-Z0-9]*) ([0-9]*.[0-9]*) ([0-9]*) ([0-9]*)'
	labels = re.findall(label_retrieval_regex, object_detections_str)
	for unicode_val, confidence, x, y in labels:
		# The 3 is added because 0, 1, and 2 are reserved for sos, eos, and pad.
		char_type = np.array(np.argwhere(sorted_charset == unicode_val)[0] + RESERVED_CHAR_COUNT) 
		label_vectors.append(char_type)

		normalized_location = np.array(
		[
			float(x)/image_width,
			float(y)/image_height
		])
		location_vectors.append(normalized_location)

		confidences.append([float(confidence)])

	return np.array(location_vectors), np.array(label_vectors), np.array(confidences)

def AddToSubmissionFile(
		image_ids, 
		original_char_locations, 
		original_char_labels,
		original_char_label_confidences,
		modified_char_locations, 
		modified_char_labels, 
		skipped_char_labels,
		skipped_char_locations,
		all_images_sizes, 
		sorted_charset, 
		output_file, 
		inclusion_confidence_threshold,
		addition_confidence_threshold):
	model_image_label_probs = np.exp(modified_char_labels)
	
	for image_index, image_id in enumerate(image_ids):
		current_image_dimensions = all_images_sizes[all_images_sizes['image_id'] == image_id]
		current_image_width = int(current_image_dimensions['image_width'])
		current_image_height = int(current_image_dimensions['image_height'])

		image_char_locations = original_char_locations[image_index]

		model_image_skipped_char_probs = np.exp(skipped_char_labels[image_index])
		char_count = len(image_char_locations) - 1 # The last char is EOS.
		char_detections = []
		for char_index in range(char_count):
			# if model_image_label_probs[image_index][char_index].max() < inclusion_confidence_threshold:
			# 	char_type_id = original_char_labels[image_index][char_index][0]
			# else:
			# 	char_type_id = torch.topk(torch.tensor(model_image_label_probs[image_index][char_index]), 1).indices
			
			original_confidence = original_char_label_confidences[image_index][char_index]
			revision_confidence = model_image_label_probs[image_index][char_index].max()
			if max(original_confidence, revision_confidence) < inclusion_confidence_threshold:
				continue

			if original_confidence > .9 or original_confidence > revision_confidence:
				char_type_id = original_char_labels[image_index][char_index][0]
			else:
				char_type_id = torch.topk(torch.tensor(model_image_label_probs[image_index][char_index]), 1).indices

			if char_type_id >= RESERVED_CHAR_COUNT:
				char_detections.append('{} {} {}'.format(
					sorted_charset[char_type_id - RESERVED_CHAR_COUNT],
					int(image_char_locations[char_index][0] * current_image_width),
					int(image_char_locations[char_index][1] * current_image_height)))

			if model_image_skipped_char_probs[char_index].max() > addition_confidence_threshold:
				skipped_char_type_id = torch.topk(torch.tensor(skipped_char_labels[image_index][char_index]), 1).indices.numpy()[0]

				delta_x = image_char_locations[char_index][0] - image_char_locations[char_index - 1][0]
				delta_y = image_char_locations[char_index][1] - image_char_locations[char_index - 1][1]
				if skipped_char_type_id >= RESERVED_CHAR_COUNT and abs(delta_x) < .05 and delta_y < .05 and delta_y > 0:
					char_detections.append('{} {} {}'.format(
						sorted_charset[skipped_char_type_id - RESERVED_CHAR_COUNT],
						int(.5 * (image_char_locations[char_index][0] + image_char_locations[char_index - 1][0]) * current_image_width),
						int(.5 * (image_char_locations[char_index][1] + image_char_locations[char_index - 1][1]) * current_image_height)))

		char_detections = ' '.join(char_detections)

		output_file.write('{},{}\n'.format(image_id, char_detections))

def CorrectSubmissionFile(
		initial_submission_file, 
		model_path,
		output_path, 
		inclusion_confidence_threshold,
		addition_confidence_threshold,
		image_sizes_path = '../Datasets/TrainingImageSizes.csv',
		sorted_charset_path = '../Datasets/SortedCharset.p'):
	# LOAD DATA.
	# The detector outputs are expected to have two columns, image_id and labels.
	raw_detector_outputs = pd.read_csv(initial_submission_file).sort_values(by = 'image_id')
	# The image sizes are expected to have three columns, image_id, image_width, and image_height.
	image_sizes = pd.read_csv(image_sizes_path).sort_values(by = 'image_id')
	# The sorted charset is just a list of U+##### strings which are in the same order as the
	# language model i/o indices.
	sorted_charset = np.array(pickle.load(open(sorted_charset_path, 'rb')))

	# CREATE THE PRE-TRAINED MODEL.
	char_label_class_count = len(sorted_charset) + RESERVED_CHAR_COUNT
	EMBEDDING_DIM_COUNT = 1024
	model = ClassificationCorrector(
		char_label_class_count = char_label_class_count, 
		embedding_dim_count = EMBEDDING_DIM_COUNT)
	model = model.cuda()
	model.eval()
	model.load_state_dict(torch.load(model_path))

	with open(output_path, 'w') as output_file:
		output_file.write('image_id,labels\n')

		# CORRECT EACH IMAGE'S CHAR DETECTOR OUTPUTS
		image_ids = np.array(raw_detector_outputs['image_id'].values)
		BATCH_SIZE = 24
		batch_count = math.ceil(len(image_ids)/BATCH_SIZE)
		# for batch_number in range(batch_count):
		batch_numbers = tqdm(
			range(batch_count), 
			leave = False, 
			total=batch_count,
			ascii = True)
		for batch_number in batch_numbers:
			# GET THE BATCH'S IMAGE IDS.
			first_image_id_index = batch_number*BATCH_SIZE
			last_image_id_index = first_image_id_index + BATCH_SIZE
			batch_image_ids = image_ids[first_image_id_index:last_image_id_index]

			# PARSE EACH IMAGE'S LABEL DETECTOR OUTPUTS. 
			batch_character_locations = []
			batch_character_labels = []
			batch_confidences = []
			for image_id in batch_image_ids:
				# GET THE IMAGE'S DIMENSIONS.
				current_image_dimensions = image_sizes[image_sizes['image_id'] == image_id]
				current_image_width = int(current_image_dimensions['image_width'])
				current_image_height = int(current_image_dimensions['image_height'])
				
				# GET THE IMAGE'S RAW DETECTOR OUTPUTS.
				current_image_raw_detector_outputs = raw_detector_outputs[raw_detector_outputs['image_id'] == image_id]
				current_image_raw_detector_labels = str(current_image_raw_detector_outputs['labels'].values[0])
				unordered_character_locations, unordered_character_labels, unordered_confidences = ObjectDetectorLabelsWithConfidenceToParallelVectors(
					current_image_raw_detector_labels, sorted_charset, current_image_width, current_image_height)
				current_image_character_locations, current_image_character_labels = GetLabelsSortedByReadOrder(
					unordered_character_locations, unordered_character_labels)
				current_image_character_locations, current_image_confidences = GetLabelsSortedByReadOrder(
					unordered_character_locations, unordered_confidences)

				# STORE THE DETECTOR OUTPUTS AS TENSORS.
				eos_char_label = [1]
				eos_location = [0, 1]
				eos_confidence = [1]
				if len(current_image_character_locations) > 0:
					batch_character_locations.append(torch.tensor(
						np.concatenate([
							current_image_character_locations, 
							[eos_location]]),
						dtype = torch.float32))
					batch_character_labels.append(torch.tensor(
						np.concatenate([
							current_image_character_labels, 
							[eos_char_label]]),
						dtype = torch.long))
					batch_confidences.append(torch.tensor(
						np.concatenate([
							current_image_confidences, 
							[eos_confidence]]),
						dtype = torch.float32))
				else:
					batch_character_locations.append(torch.tensor([eos_location], dtype = torch.float32))
					batch_character_labels.append(torch.tensor([eos_char_label], dtype = torch.long))
					batch_confidences.append(torch.tensor([eos_confidence], dtype = torch.float32))

			# PAD THE CHARACTER DETECTIONS.
			max_sequence_length = max(len(sequence) for sequence in batch_character_locations)
			current_batch_size = len(batch_character_locations)
			padded_batch_char_locations = torch.zeros(
				size = (max_sequence_length, current_batch_size, 2),
				device=DEVICE)
			padded_batch_char_labels = torch.ones(
				size = (max_sequence_length, current_batch_size, 1),
				dtype = torch.long,
				device = DEVICE)
			padded_batch_confidences = torch.ones(
				size = (max_sequence_length, current_batch_size, 1),
				dtype = torch.float32,
				device = DEVICE)
			for image_index in range(current_batch_size):
				current_image_character_locations = batch_character_locations[image_index]
				current_image_label_count = len(current_image_character_locations)
				padded_batch_char_locations[0:current_image_label_count, image_index,] = current_image_character_locations

				current_image_character_labels = batch_character_labels[image_index]
				padded_batch_char_labels[0:current_image_label_count, image_index,] = current_image_character_labels

				current_image_confidences = batch_confidences[image_index]
				padded_batch_confidences[0:current_image_label_count, image_index,] = current_image_confidences

			# USE THE MODEL TO CORRECT THE DETECTOR OUTPUTS.
			with torch.no_grad():
				model_output_labels, model_output_locations, skipped_char_labels, skipped_char_locations = model(
					input_character_locations = padded_batch_char_locations, 
					input_character_labels = padded_batch_char_labels,
					input_character_confidences = padded_batch_confidences)
			AddToSubmissionFile(
				image_ids = batch_image_ids, 
				original_char_locations = [v.numpy() for v in batch_character_locations], 
				original_char_labels = [v.numpy() for v in batch_character_labels],
				original_char_label_confidences = [v.numpy() for v in batch_confidences],
				modified_char_locations = model_output_locations.cpu().transpose(0,1).numpy(), 
				modified_char_labels = model_output_labels.cpu().transpose(0,1).numpy(), 
				skipped_char_labels = skipped_char_labels.cpu().transpose(0,1).numpy(),
				skipped_char_locations = skipped_char_locations.cpu().transpose(0,1).numpy(),
				all_images_sizes = image_sizes, 
				sorted_charset = sorted_charset, 
				output_file = output_file,
				inclusion_confidence_threshold = inclusion_confidence_threshold,
				addition_confidence_threshold = addition_confidence_threshold)

if __name__ == '__main__':
	# CorrectSubmissionFile(
	# 	# initial_submission_file = 'RawSubmissionFiles/CrossValidationSetPredictions_75thresh.csv',
	# 	initial_submission_file = 'RawSubmissionFiles/WithConfidence/CrossValidationSetPredictions_SubsetModelWithConfidence.csv',
	# 	model_path = 'TempModel_12layerBackup.pth',
	# 	output_path = 'CorrectedSubmissionFile.csv',
	# 	inclusion_confidence_threshold = float(sys.argv[1]),
	# 	addition_confidence_threshold = float(sys.argv[2]))

	CorrectSubmissionFile(
		initial_submission_file = 'RawSubmissionFiles/WithConfidence/FinalTestSetPredictions_FinalModelWithConfidence.csv',
		model_path = 'TempModel_12layerBackup.pth',
		output_path = 'CorrectedSubmissionFile.csv',
		image_sizes_path = '../Datasets/FinalTestImageSizes.csv',
		inclusion_confidence_threshold = float(sys.argv[1]),
		addition_confidence_threshold = float(sys.argv[2]))