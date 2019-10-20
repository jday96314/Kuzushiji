import re
import numpy as np
import pandas as pd
import pickle
import torch
import random
import PIL
from tqdm import tqdm
from PIL import ImageDraw, ImageFont
from Model import *
from CharSequencesDataset import *
from ErrorStats import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.cluster import DBSCAN


DEVICE = torch.device("cuda")

def VisualizeCharacterReadOrder(image_id, ordered_points):
	image = PIL.Image.open('../Datasets/train_images/' + image_id + '.jpg')
	image_width, image_height = image.size
	image_editor = ImageDraw.Draw(image)
	font = ImageFont.truetype("../Datasets/abel-regular.ttf", 50)
	for index, (normalized_x, normalized_y) in enumerate(ordered_points):
		image_editor.text(
			(normalized_x * image_width, normalized_y * image_height), 
			str(index), 
			(255, 0, 0), 
			font = font)
	image = image.resize((image_width // 4, image_height//4), resample = PIL.Image.BICUBIC)
	image.save('../ImageReadOrders/' + image_id + '.jpg')

# def GetCharSequences(image_sizes, unordered_ground_truth_sequences, sorted_charset):
# 	all_sequences = []
# 	for image_index in range(1000):#len(image_sizes)):
# 		image_width, image_height = image_sizes[image_index]
# 		ground_truth_image_labels = unordered_ground_truth_sequences[image_index]

# 		image_detected_objects = ObjectDetectorLabelsToVector(
# 			ground_truth_image_labels, sorted_charset, image_width, image_height)

# 		image_detected_objects_in_read_order = GetLabelsSortedByReadOrder(image_detected_objects)

# 		if len(image_detected_objects_in_read_order) > 0:
# 			all_sequences.append(torch.tensor(image_detected_objects_in_read_order))
# 	return nn.utils.rnn.pad_sequence(all_sequences)

def PadSequences(sequence_tuples):
	max_seq_len = 0
	for sequence_tuple in sequence_tuples:
		max_seq_len = max(max_seq_len, max(sequence_tuple[0].shape[0], sequence_tuple[0].shape[0]))

	# CREATE THE PADED SEQUENCES.
	padded_sequence_a_locations = torch.zeros(max_seq_len, len(sequence_tuples), 2)
	pad_char_label = 2
	padded_sequence_a_labels = torch.ones(
		size = (max_seq_len, len(sequence_tuples), 1),
		dtype = torch.long) * pad_char_label
	padded_sequence_b_locations = torch.zeros(max_seq_len, len(sequence_tuples), 2)
	padded_sequence_b_labels = torch.ones(
		size = (max_seq_len, len(sequence_tuples), 1),
		dtype = torch.long) * pad_char_label

	for sequence_index, sub_sequences_tuple in enumerate(sequence_tuples):
		sequence_a_locations = sub_sequences_tuple[0]
		padded_sequence_a_locations[0:len(sequence_a_locations), sequence_index,] = sequence_a_locations
		sequence_a_labels = sub_sequences_tuple[1]
		padded_sequence_a_labels[0:len(sequence_a_labels), sequence_index,] = sequence_a_labels

		sequence_b_locations = sub_sequences_tuple[2]
		padded_sequence_b_locations[0:len(sequence_b_locations), sequence_index,] = sequence_b_locations
		sequence_b_labels = sub_sequences_tuple[3]
		padded_sequence_b_labels[0:len(sequence_b_labels), sequence_index,] = sequence_b_labels

	return padded_sequence_a_locations, padded_sequence_a_labels, padded_sequence_b_locations, padded_sequence_b_labels

if __name__ == '__main__':
	# LOAD THE DATASET.
	PERFECT_PREDICTIONS_PATH = '../Datasets/PerfectObjectDetectorPredictions.csv'
	perfect_char_labels = pd.read_csv(PERFECT_PREDICTIONS_PATH).sort_values(by = 'image_id')
	testing_object_detector_predictions = pd.read_csv('../Datasets/TestingObjectDetectorPredictions.csv').sort_values(by = 'image_id')
	training_image_perfect_detector_predictions = perfect_char_labels[
	 	~perfect_char_labels['image_id'].isin(testing_object_detector_predictions['image_id'])]
	training_image_ids = training_image_perfect_detector_predictions['image_id'].values
	testing_image_ids = testing_object_detector_predictions['image_id'].values
	sorted_charset = np.array(pickle.load(open('../Datasets/SortedCharset.p', 'rb')))

	IMAGE_SIZES_PATH = '../Datasets/TrainingImageSizes.csv'
	seqnece_mangler = ObjectDetectionsMangler(
		normalized_confusion_matrix = None, 
		label_location_x_stddev = None,
		label_location_y_stddev = None,
		mean_extraneous_characters_per_image = None,
		stddev_extraneous_characters_per_image = None,
		extraneous_character_type_probs = None,
		character_omission_probability = None)
	training_dataset = FaceLandmarksDataset(
		PERFECT_PREDICTIONS_PATH,
		IMAGE_SIZES_PATH,
		training_image_ids,
		sorted_charset,
		seqnece_mangler)
	BATCH_SIZE = 64
	# Todo: Investigate pinning memory.
	training_data_loader = DataLoader(
		training_dataset, 
		batch_size = BATCH_SIZE, 
		shuffle = True, 
		num_workers=4,
		collate_fn = PadSequences,
		pin_memory = True)	

	# CREATE THE MODELS.
	# 3 is added to reserve space for eos, sos, and pad
	char_label_class_count = len(sorted_charset) + 3
	EMBEDDING_DIM_COUNT = 256
	RECURRENT_UNIT_COUNT = 512
	encoder = Encoder(
		char_class_count = char_label_class_count, 
		embedding_dim_count = EMBEDDING_DIM_COUNT,
		recurrent_unit_count = RECURRENT_UNIT_COUNT)
	encoder = encoder.to(DEVICE)
	encoder_optimizer = optim.SGD(encoder.parameters(), 1e-8, momentum = .9)

	MAX_CHARACTERS_PER_PAGE = 1000
	decoder = Decoder(
		recurrent_unit_count = RECURRENT_UNIT_COUNT, 
		embedding_dim_count = EMBEDDING_DIM_COUNT,
		char_label_class_count = char_label_class_count, 
		dropout_rate = 0,
		max_sequence_length = MAX_CHARACTERS_PER_PAGE)
	decoder.to(DEVICE)
	decoder_optimizer = optim.SGD(decoder.parameters(), 1e-8, momentum = .9)

	# DEFINE THE LOSS FUNCTION.
	classification_loss_function = nn.NLLLoss(reduction = 'none').to(DEVICE)
	regression_loss_function = nn.MSELoss(reduction = 'none').to(DEVICE)

	# TRAIN THE MODELS.
	EPOCH_COUNT = 10
	for epoch in range(EPOCH_COUNT):
		#print(, end = ' ')
		epoch_batches = tqdm(
			iter(training_data_loader), 
			leave = False, 
			total=len(training_data_loader),
			ascii = True,
			desc = 'Epoch {}/{}'.format(epoch + 1, EPOCH_COUNT))
		encoder.train()
		decoder.train()
		training_batch_classification_losses = []
		training_batch_regression_losses = []
		for batch in epoch_batches:
			ground_truth_location_sequence_batch = batch[0].to(DEVICE)
			ground_truth_label_sequence_batch = batch[1].to(DEVICE)
			mangled_location_sequence_batch = batch[2].to(DEVICE)
			mangled_label_sequence_batch = batch[3].to(DEVICE)
			sequences_length = mangled_label_sequence_batch.shape[0]

			# FORWARD PASS.
			encoder.zero_grad()
			decoder.zero_grad()

			current_batch_size = mangled_label_sequence_batch.shape[1]
			encoder_hidden = encoder.initHidden(current_batch_size).to(DEVICE)
			encoder_outputs, encoder_hidden = encoder(
				mangled_location_sequence_batch, 
				mangled_label_sequence_batch,
				encoder_hidden)

			decoder_input_char_location = torch.tensor([[0, 1]] * current_batch_size, device = DEVICE, dtype = torch.float32)
			decoder_input_char_label = torch.tensor([[0]] * current_batch_size, device = DEVICE, dtype = torch.long)
			decoder_hidden = encoder_hidden
			total_classification_loss = 0
			total_regression_loss = 0
			total_matching_label_count = 0
			total_label_count = 0
			for char_index in range(sequences_length):
				# COMPUTE THE LOSS.
				decoded_char_locations, decoded_char_labels, decoder_hidden = decoder(
					decoder_input_char_location,
					decoder_input_char_label, 
					decoder_hidden, 
					encoder_outputs)
				decoder_input_char_location = ground_truth_location_sequence_batch[char_index]
				decoder_input_char_label = ground_truth_label_sequence_batch[char_index]

				flattenend_ground_truth_labels = ground_truth_label_sequence_batch[char_index].view(-1)
				loss_mask = (flattenend_ground_truth_labels != 2).float()
				chars_classification_losses = classification_loss_function(
					decoded_char_labels,
					flattenend_ground_truth_labels)
				total_classification_loss += torch.sum(chars_classification_losses * loss_mask)
				chars_regression_losses = regression_loss_function(
					decoded_char_locations,
					ground_truth_location_sequence_batch[char_index].view(current_batch_size, -1))
				total_regression_loss += torch.sum(chars_regression_losses)

				# COMPUTE THE ACCURACY METRIC.
				cpu_ground_truth_labels = flattenend_ground_truth_labels.cpu().numpy()
				cpu_detected_labels = decoded_char_labels.detach().cpu().numpy()
				cpu_detected_labels = np.argmax(cpu_detected_labels, 1)
				matching_label_count = sum((cpu_ground_truth_labels == cpu_detected_labels) * loss_mask.cpu().numpy())
				total_matching_label_count += matching_label_count
				unpadded_label_count = sum(loss_mask.cpu().numpy())
				total_label_count += unpadded_label_count

			# BACKWARD PASS.
			NORMALIZATION_FACTOR = .05
			total_loss = NORMALIZATION_FACTOR*total_classification_loss + total_regression_loss
			total_loss.backward()
			
			# WEIGHT UPDATES.
			encoder_optimizer.step()
			decoder_optimizer.step()

			# RECORD THE LOSSES.
			training_batch_classification_losses.append(total_classification_loss.detach().cpu().numpy())
			training_batch_regression_losses.append(total_regression_loss.detach().cpu().numpy())
		
		print('Epoch {}/{} loss - tc: {:.1f}, tr: {:.1f}, ta: {:.5f}'.format(
			epoch + 1, 
			EPOCH_COUNT,
			np.mean(training_batch_classification_losses),
			np.mean(training_batch_regression_losses),
			total_matching_label_count/total_label_count))
		print(cpu_detected_labels)
		print(cpu_ground_truth_labels)
		