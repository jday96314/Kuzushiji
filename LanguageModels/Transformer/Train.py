import random
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
from DataMangler import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

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

def PadSequences(sequence_tuples):
	# # debug, please delete.
	# chars_to_keep = 5
	# sequence_tuples = [(a[:chars_to_keep],b[:chars_to_keep],c[:chars_to_keep],d[:chars_to_keep]) for a,b,c,d in sequence_tuples]

	max_seq_len = 0
	unpadded_sequence_a_lengths = torch.zeros(len(sequence_tuples), dtype=torch.int32)
	unpadded_sequence_b_lengths = torch.zeros(len(sequence_tuples), dtype=torch.int32)
	for sequence_index, sequence_tuple in enumerate(sequence_tuples):
		max_seq_len = max(max_seq_len, max(sequence_tuple[0].shape[0], sequence_tuple[3].shape[0]))
		unpadded_sequence_a_lengths[sequence_index] = sequence_tuple[0].shape[0]
		unpadded_sequence_b_lengths[sequence_index] = sequence_tuple[3].shape[0]

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

	return (
		padded_sequence_a_locations, 
		padded_sequence_a_labels, 
		unpadded_sequence_a_lengths,
		padded_sequence_b_locations, 
		padded_sequence_b_labels,
		unpadded_sequence_b_lengths)

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
	testing_dataset = FaceLandmarksDataset(
		PERFECT_PREDICTIONS_PATH,
		IMAGE_SIZES_PATH,
		testing_image_ids,
		sorted_charset,
		seqnece_mangler)
	BATCH_SIZE = 4
	# Todo: Investigate pinning memory.
	training_data_loader = DataLoader(
		training_dataset, 
		batch_size = BATCH_SIZE, 
		shuffle = True, 
		num_workers=4,
		collate_fn = PadSequences,
		pin_memory = True)	
	testing_data_loader = DataLoader(
		testing_dataset, 
		batch_size = BATCH_SIZE*2, 
		shuffle = True, 
		num_workers=4,
		collate_fn = PadSequences,
		pin_memory = True)	

	# CREATE THE MODELS.
	# 3 is added to reserve space for eos, sos, and pad
	char_label_class_count = len(sorted_charset) + 3
	EMBEDDING_DIM_COUNT = 1024
	model = NextCharacterPredictor(
		char_label_class_count = char_label_class_count, 
		embedding_dim_count = EMBEDDING_DIM_COUNT)
	model = model.to(DEVICE)
	#optimizer = optim.SGD(model.parameters(), 1e-4, momentum=.9)
	optimizer = optim.SGD(model.parameters(), .05)
	#optimizer = optim.Adam(model.parameters(), 5e-4)

	# # LOAD THE MODEL'S PRE-TRAINED PARAMETERS.
	# model.load_state_dict(torch.load('TempModel_Modified.pth'))

	# DEFINE THE LOSS FUNCTION.
	regression_loss_function = nn.MSELoss().to(DEVICE)
	classification_loss_function = nn.NLLLoss().to(DEVICE)

	# TRAIN THE MODELS.
	EPOCH_COUNT = 50
	for epoch in range(EPOCH_COUNT):
		epoch_training_batches = tqdm(
			iter(training_data_loader), 
			leave = False, 
			total=len(training_data_loader),
			ascii = True,
			desc = 'Epoch {}/{}'.format(epoch + 1, EPOCH_COUNT))
		model.train()
		training_batch_classification_losses = []
		training_batch_next_regression_losses = []
		training_batch_next_classification_losses = []
		for batch in epoch_training_batches:
			# EXTRACT THE BATCH'S DATA.
			ground_truth_location_sequence_batch = batch[0].to(DEVICE)
			ground_truth_label_sequence_batch = batch[1].to(DEVICE)
			unpadded_ground_truth_sequence_lengths_batch = batch[2].to(DEVICE)
			mangled_location_sequence_batch = batch[3].to(DEVICE)
			mangled_label_sequence_batch = batch[4].to(DEVICE)
			unpadded_mangled_sequence_lengths_batch = batch[5].to(DEVICE)
			
			current_batch_size = mangled_label_sequence_batch.shape[1]

			# RESET THE GRADIENTS.
			model.zero_grad()

			# CORRECT THE SEQUENCES.
			model_output_labels, model_output_next_char_locations, model_output_next_char_labels = model(
				input_character_locations = mangled_location_sequence_batch, 
				input_character_labels = mangled_label_sequence_batch)

			# COMPUTE THE LOSS.
			main_classification_loss = classification_loss_function(
				model_output_labels.view(len(ground_truth_label_sequence_batch) * current_batch_size, -1),
				ground_truth_label_sequence_batch.view(-1))
			next_regression_loss = regression_loss_function(
				model_output_next_char_locations[:-1], 
				ground_truth_location_sequence_batch[1:])
			next_classification_loss = classification_loss_function(
				model_output_next_char_labels[:-1].view(
					(len(ground_truth_label_sequence_batch) - 1) * current_batch_size, -1),
				ground_truth_label_sequence_batch[1:].view(-1))

			# UPDATE THE NETWORK.
			next_regression_loss_weight = max(.0001, -.5*epoch + 5) # max(.8**epoch, -.5*epoch + 5)
			next_classification_loss_weight = .9**epoch
			total_loss = (1 - next_classification_loss_weight) * main_classification_loss
			total_loss += next_regression_loss_weight*next_regression_loss 
			total_loss += next_classification_loss_weight*next_classification_loss
			total_loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), .5)
			optimizer.step()

			# RECORD THE LOSSES.
			training_batch_classification_losses.append(main_classification_loss.detach().cpu().numpy())
			training_batch_next_regression_losses.append(next_regression_loss.detach().cpu().numpy())
			training_batch_next_classification_losses.append(next_classification_loss.detach().cpu().numpy())
		
		epoch_testing_batches = tqdm(
			iter(testing_data_loader), 
			leave = False, 
			total=len(testing_data_loader),
			ascii = True,
			desc = 'Cross validating model'.format(epoch + 1, EPOCH_COUNT))
		model.eval()
		testing_batch_classification_losses = []
		testing_batch_next_regression_losses = []
		testing_batch_next_classification_losses = []
		total_distance_from_real_points = 0
		total_correct_character_label_count = 0
		total_correct_next_character_label_count = 0
		total_real_character_count = 0
		with torch.no_grad():
			for batch in epoch_testing_batches:
				# EXTRACT THE BATCH'S DATA.
				ground_truth_location_sequence_batch = batch[0].to(DEVICE)
				ground_truth_label_sequence_batch = batch[1].to(DEVICE)
				unpadded_ground_truth_sequence_lengths_batch = batch[2].to(DEVICE)
				mangled_location_sequence_batch = batch[3].to(DEVICE)
				mangled_label_sequence_batch = batch[4].to(DEVICE)
				unpadded_mangled_sequence_lengths_batch = batch[5].to(DEVICE)
				
				current_batch_size = mangled_label_sequence_batch.shape[1]

				# CORRECT THE SEQUENCES.
				model_output_labels, model_output_next_char_locations, model_output_next_char_labels = model(
					input_character_locations = mangled_location_sequence_batch, 
					input_character_labels = mangled_label_sequence_batch)

				# COMPUTE THE LOSS.
				main_classification_loss = classification_loss_function(
					model_output_labels.view(len(ground_truth_label_sequence_batch) * current_batch_size, -1),
					ground_truth_label_sequence_batch.view(-1))
				next_regression_loss = regression_loss_function(
					model_output_next_char_locations[:-1], 
					ground_truth_location_sequence_batch[1:])
				next_classification_loss = classification_loss_function(
					model_output_next_char_labels[:-1].view(
						(len(ground_truth_label_sequence_batch) - 1) * current_batch_size, -1),
					ground_truth_label_sequence_batch[1:].view(-1))

				# COMPUTE THE PERFORMANCE METRICS.
				correctly_labeled_flags = (ground_truth_label_sequence_batch == torch.topk(model_output_labels, 1).indices).int()
				real_character_flags = (ground_truth_label_sequence_batch != 2).int()
				total_correct_character_label_count += sum((correctly_labeled_flags * real_character_flags).view(-1).cpu().numpy())
				total_real_character_count += sum(real_character_flags.view(-1).cpu().numpy())

				predicted_location_errors = (model_output_next_char_locations[:-1] - ground_truth_location_sequence_batch[1:])
				predicted_location_error_distances = torch.sqrt(predicted_location_errors[:,:,0]**2 + predicted_location_errors[:,:,1]**2)
				total_distance_from_real_points += sum((
					(predicted_location_error_distances * real_character_flags[1:].view(predicted_location_error_distances.shape).float()).view(-1).cpu().numpy()))

				correctly_labeled_flags = (ground_truth_label_sequence_batch[1:] == torch.topk(model_output_next_char_labels[:-1], 1).indices).int()
				real_character_flags = (ground_truth_label_sequence_batch[1:] != 2).int()
				total_correct_next_character_label_count += sum((correctly_labeled_flags * real_character_flags).view(-1).cpu().numpy())
			
				# RECORD THE LOSSES.
				testing_batch_classification_losses.append(main_classification_loss.detach().cpu().numpy())
				testing_batch_next_regression_losses.append(next_regression_loss.detach().cpu().numpy())
				testing_batch_next_classification_losses.append(next_classification_loss.detach().cpu().numpy())

		# DISPLAY THE MODEL'S PERFORMANCE STATISTICS.
		# tr_c = training current character classification loss
		# tr_nr = training next character location regression loss
		# tr_nc = training next character classification loss
		# te_c = testing current character classification loss
		# te_nr = testing next character location regression loss
		# te_nc = testing next character classification loss
		# te_acc = testing current character classification accuracy
 		# te_n_mu_d = testing mean distance between next character's predicted location and its actual location (normalized for image size).
		# te_n_acc = testing next character classification accuracy
		info_format = 'Epoch {}/{} loss - tr_c: {:.3f}, tr_nr: {:.3f}, tr_nc: {:.3f}, te_c: {:.3f}, te_nr: {:.3f}, te_nc: {:.3f}, te_acc: {:.2f}%, te_n_mu_d: {:.2f}, te_n_acc: {:.2f}%'
		print(info_format.format(
			epoch + 1, 
			EPOCH_COUNT,
			np.mean(training_batch_classification_losses),
			np.mean(training_batch_next_regression_losses),
			np.mean(training_batch_next_classification_losses),
			np.mean(testing_batch_classification_losses),
			np.mean(testing_batch_next_regression_losses),
			np.mean(testing_batch_next_classification_losses),
			100*total_correct_character_label_count/total_real_character_count,
			total_distance_from_real_points/total_real_character_count,
			total_correct_next_character_label_count/total_real_character_count))

		torch.save(model.state_dict(), 'TempModel.pth')
		