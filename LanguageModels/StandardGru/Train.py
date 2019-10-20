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
# For char ords are reserved for the following:
# 0 = sos (start of string)
# 1 = eos (end of string)
# 2 = pad (padding)
# 3 = NA (Not a real char)
RESERVED_CHAR_COUNT = 4

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
	max_seq_len = 0
	unpadded_sequence_a_lengths = torch.zeros(len(sequence_tuples), dtype=torch.int32)
	unpadded_sequence_b_lengths = torch.zeros(len(sequence_tuples), dtype=torch.int32)
	unpadded_sequence_c_lengths = torch.zeros(len(sequence_tuples), dtype=torch.int32)
	for sequence_index, sequence_tuple in enumerate(sequence_tuples):
		max_seq_len = max(max_seq_len, max(sequence_tuple[0].shape[0], sequence_tuple[2].shape[0], sequence_tuple[4].shape[0]))
		unpadded_sequence_a_lengths[sequence_index] = sequence_tuple[0].shape[0]
		unpadded_sequence_b_lengths[sequence_index] = sequence_tuple[2].shape[0]
		unpadded_sequence_c_lengths[sequence_index] = sequence_tuple[4].shape[0]

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
	padded_sequence_c_locations = torch.zeros(max_seq_len, len(sequence_tuples), 2)
	padded_sequence_c_labels = torch.ones(
		size = (max_seq_len, len(sequence_tuples), 1),
		dtype = torch.long) * pad_char_label
	padded_sequence_c_confidences = torch.zeros(max_seq_len, len(sequence_tuples), 1)

	for sequence_index, sub_sequences_tuple in enumerate(sequence_tuples):
		sequence_a_locations = sub_sequences_tuple[0]
		padded_sequence_a_locations[0:len(sequence_a_locations), sequence_index,] = sequence_a_locations
		sequence_a_labels = sub_sequences_tuple[1]
		padded_sequence_a_labels[0:len(sequence_a_labels), sequence_index,] = sequence_a_labels

		sequence_b_locations = sub_sequences_tuple[2]
		padded_sequence_b_locations[0:len(sequence_b_locations), sequence_index,] = sequence_b_locations
		sequence_b_labels = sub_sequences_tuple[3]
		padded_sequence_b_labels[0:len(sequence_b_labels), sequence_index,] = sequence_b_labels

		sequence_c_locations = sub_sequences_tuple[4]
		padded_sequence_c_locations[0:len(sequence_c_locations), sequence_index,] = sequence_c_locations
		sequence_c_labels = sub_sequences_tuple[5]
		padded_sequence_c_labels[0:len(sequence_c_labels), sequence_index,] = sequence_c_labels
		sequence_c_confidences = sub_sequences_tuple[6]
		padded_sequence_c_confidences[0:len(sequence_c_locations), sequence_index,] = sequence_c_confidences

	return (
		padded_sequence_a_locations, 
		padded_sequence_a_labels, 
		unpadded_sequence_a_lengths,
		padded_sequence_b_locations, 
		padded_sequence_b_labels,
		unpadded_sequence_b_lengths,
		padded_sequence_c_locations, 
		padded_sequence_c_labels,
		padded_sequence_c_confidences,
		unpadded_sequence_c_lengths)

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
	ERROR_STATS_PATH = '../Datasets/ErrorStats.p'
	error_stats = pickle.load(open(ERROR_STATS_PATH, 'rb'))
	EXTRANEOUS_DETECTION_CHAR_LABEL_ID = 3
	training_seqnece_mangler = ObjectDetectionsMangler(
		error_stats,
		RESERVED_CHAR_COUNT,
		EXTRANEOUS_DETECTION_CHAR_LABEL_ID,
		boost_drop_probs=True)
	training_dataset = CharSequencesDataset(
		PERFECT_PREDICTIONS_PATH,
		IMAGE_SIZES_PATH,
		training_image_ids,
		sorted_charset,
		training_seqnece_mangler,
		max_sequence_length = 150)
	testing_seqnece_mangler = ObjectDetectionsMangler(
		error_stats,
		RESERVED_CHAR_COUNT,
		EXTRANEOUS_DETECTION_CHAR_LABEL_ID,
		boost_drop_probs=False)
	testing_dataset = CharSequencesDataset(
		PERFECT_PREDICTIONS_PATH,
		IMAGE_SIZES_PATH,
		testing_image_ids,
		sorted_charset,
		testing_seqnece_mangler,
		max_sequence_length = 200)
	BATCH_SIZE = 16
	# Todo: Investigate pinning memory.
	training_data_loader = DataLoader(
		training_dataset, 
		batch_size = BATCH_SIZE, 
		shuffle = True, 
		num_workers=8,
		collate_fn = PadSequences,
		pin_memory = True)	
	testing_data_loader = DataLoader(
		testing_dataset, 
		batch_size = BATCH_SIZE*2, 
		shuffle = True, 
		num_workers=8,
		collate_fn = PadSequences,
		pin_memory = True)	

	# CREATE THE MODELS.
	# 3 is added to reserve space for eos, sos, and pad
	char_label_class_count = len(sorted_charset) + RESERVED_CHAR_COUNT
	model = ClassificationCorrector(
		char_label_class_count = char_label_class_count, 
		embedding_dim_count = 1024)
	model = model.to(DEVICE)
	optimizer = optim.SGD(model.parameters(), .2, momentum=.9, nesterov=True)
	learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(
		optimizer, 
		milestones = [30],
		gamma = .5)

	# LOAD THE MODEL'S PRE-TRAINED PARAMETERS.
	model.load_state_dict(torch.load('TempModel_12layerBackup.pth'))

	# DEFINE THE LOSS FUNCTION.
	regression_loss_function = nn.MSELoss().to(DEVICE)
	classification_loss_function = nn.NLLLoss().to(DEVICE)
	skipped_char_classification_loss_function = nn.NLLLoss(
		#weight=torch.tensor([.1 for i in range(RESERVED_CHAR_COUNT)] + [1 for i in range(len(sorted_charset))])
	).to(DEVICE)
	skipped_char_regression_loss_function = nn.SmoothL1Loss(reduction = 'none').to(DEVICE)

	# TRAIN THE MODELS.
	EPOCH_COUNT = 150
	for epoch in range(EPOCH_COUNT):
		epoch_training_batches = tqdm(
			iter(training_data_loader), 
			leave = False, 
			total=len(training_data_loader),
			ascii = True,
			desc = 'Epoch {}/{}'.format(epoch + 1, EPOCH_COUNT))
		model.train()
		training_batch_classification_losses = []
		training_batch_regression_losses = []
		training_batch_skipped_char_classification_losses = []
		training_batch_skipped_char_regression_losses = []
		for batch in epoch_training_batches:
			# EXTRACT THE BATCH'S DATA.
			ground_truth_location_sequence_batch = batch[0].to(DEVICE)
			ground_truth_label_sequence_batch = batch[1].to(DEVICE)
			#unpadded_ground_truth_sequence_lengths_batch = batch[2].to(DEVICE)

			skipped_char_location_sequence_batch = batch[3].to(DEVICE)
			skipped_char_label_sequence_batch = batch[4].to(DEVICE)
			#unpadded_skipped_char_serquence_lengths_batch = batch[5].to(DEVICE)

			mangled_location_sequence_batch = batch[6].to(DEVICE)
			mangled_label_sequence_batch = batch[7].to(DEVICE)
			mangled_label_confidences_batch = batch[8].to(DEVICE)
			#unpadded_mangled_sequence_lengths_batch = batch[9].to(DEVICE)

			current_batch_size = mangled_label_sequence_batch.shape[1]

			# RESET THE GRADIENTS.
			model.zero_grad()

			# CORRECT THE SEQUENCES.
			model_output_labels, model_output_char_locations, predicted_skipped_char_labels, predicted_skipped_char_locations = model(
				input_character_locations = mangled_location_sequence_batch, 
				input_character_labels = mangled_label_sequence_batch,
				input_character_confidences = mangled_label_confidences_batch)

			# COMPUTE THE LOSS.
			classification_loss = classification_loss_function(
				model_output_labels.view(len(ground_truth_label_sequence_batch) * current_batch_size, -1),
				ground_truth_label_sequence_batch.view(-1))
			regression_loss = regression_loss_function(
				model_output_char_locations, 
				ground_truth_location_sequence_batch)
			skipped_char_classification_loss = skipped_char_classification_loss_function(
				predicted_skipped_char_labels.view(len(ground_truth_label_sequence_batch) * current_batch_size, -1),
				skipped_char_label_sequence_batch.view(-1))
			skipped_char_regression_loss = skipped_char_regression_loss_function(
				predicted_skipped_char_locations, 
				skipped_char_location_sequence_batch)

			skipped_char_regression_loss_mask = (skipped_char_label_sequence_batch >= RESERVED_CHAR_COUNT).float().cuda()
			skipped_char_regression_loss = torch.sum(skipped_char_regression_loss * skipped_char_regression_loss_mask) / (1e-6 + torch.sum(skipped_char_regression_loss_mask))

			#print(torch.sum(skipped_char_regression_loss_mask))

			# UPDATE THE NETWORK.
			total_loss = classification_loss
			regression_loss_weight = max(.0001, -.5*epoch + 3)
			total_loss += regression_loss_weight*regression_loss 
			total_loss += skipped_char_classification_loss
			total_loss += 5 * skipped_char_regression_loss
			total_loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), .5)
			optimizer.step()

			# RECORD THE LOSSES.
			training_batch_classification_losses.append(classification_loss.detach().cpu().numpy())
			training_batch_regression_losses.append(regression_loss.detach().cpu().numpy())
			training_batch_skipped_char_classification_losses.append(skipped_char_classification_loss.detach().cpu().numpy())
			training_batch_skipped_char_regression_losses.append(skipped_char_regression_loss.detach().cpu().numpy())
		
		learning_rate_scheduler.step()

		epoch_testing_batches = tqdm(
			iter(testing_data_loader), 
			leave = False, 
			total=len(testing_data_loader),
			ascii = True,
			desc = 'Cross validating model'.format(epoch + 1, EPOCH_COUNT))
		model.eval()
		testing_batch_classification_losses = []
		testing_batch_regression_losses = []
		testing_batch_skipped_char_classification_losses = []
		testing_batch_skipped_char_regression_losses = []
		total_distance_from_real_points = 0
		total_correct_character_label_count = 0
		total_real_character_count = 0
		total_correct_skipped_char_label_count = 0
		total_real_skipped_char_count = 0
		total_distance_from_real_skipped_chars = 0
		total_correct_unfiltered_skipped_char_sequence_outputs = 0
		with torch.no_grad():
			for batch in epoch_testing_batches:
				# EXTRACT THE BATCH'S DATA.
				ground_truth_location_sequence_batch = batch[0].to(DEVICE)
				ground_truth_label_sequence_batch = batch[1].to(DEVICE)
				#unpadded_ground_truth_sequence_lengths_batch = batch[2].to(DEVICE)

				skipped_char_location_sequence_batch = batch[3].to(DEVICE)
				skipped_char_label_sequence_batch = batch[4].to(DEVICE)
				#unpadded_skipped_char_serquence_lengths_batch = batch[5].to(DEVICE)

				mangled_location_sequence_batch = batch[6].to(DEVICE)
				mangled_label_sequence_batch = batch[7].to(DEVICE)
				mangled_label_confidences_batch = batch[8].to(DEVICE)
				#unpadded_mangled_sequence_lengths_batch = batch[8].to(DEVICE)

				current_batch_size = mangled_label_sequence_batch.shape[1]

				# CORRECT THE SEQUENCES.
				model_output_labels, model_output_char_locations, predicted_skipped_char_labels, predicted_skipped_char_locations = model(
					input_character_locations = mangled_location_sequence_batch, 
					input_character_labels = mangled_label_sequence_batch,
					input_character_confidences = mangled_label_confidences_batch)

				# COMPUTE THE LOSS.
				classification_loss = classification_loss_function(
					model_output_labels.view(len(ground_truth_label_sequence_batch) * current_batch_size, -1),
					ground_truth_label_sequence_batch.view(-1))
				regression_loss = regression_loss_function(
					model_output_char_locations, 
					ground_truth_location_sequence_batch)
				skipped_char_classification_loss = skipped_char_classification_loss_function(
					predicted_skipped_char_labels.view(len(ground_truth_label_sequence_batch) * current_batch_size, -1),
					skipped_char_label_sequence_batch.view(-1))
				skipped_char_regression_loss = skipped_char_regression_loss_function(
					predicted_skipped_char_locations, 
					skipped_char_location_sequence_batch)

				skipped_char_regression_loss_mask = (skipped_char_label_sequence_batch >= RESERVED_CHAR_COUNT).float().cuda()
				skipped_char_regression_loss = torch.sum(skipped_char_regression_loss * skipped_char_regression_loss_mask) / torch.sum(skipped_char_regression_loss_mask)

				# COMPUTE THE PERFORMANCE METRICS.
				correctly_labeled_flags = (ground_truth_label_sequence_batch == torch.topk(model_output_labels, 1).indices).int()
				real_character_flags = (ground_truth_label_sequence_batch >= RESERVED_CHAR_COUNT).int()
				total_correct_character_label_count += sum((correctly_labeled_flags * real_character_flags).view(-1).cpu().numpy())
				total_real_character_count += sum(real_character_flags.view(-1).cpu().numpy())

				predicted_location_errors = (model_output_char_locations - ground_truth_location_sequence_batch)
				predicted_location_error_distances = torch.sqrt(predicted_location_errors[:,:,0]**2 + predicted_location_errors[:,:,1]**2)
				total_distance_from_real_points += sum((
					(predicted_location_error_distances * real_character_flags.view(predicted_location_error_distances.shape).float()).view(-1).cpu().numpy()))

				correctly_labeled_skipped_char_flags = (skipped_char_label_sequence_batch == torch.topk(predicted_skipped_char_labels, 1).indices).int()
				real_skipped_char_flags = (skipped_char_label_sequence_batch >= RESERVED_CHAR_COUNT).int()
				total_correct_skipped_char_label_count += sum((correctly_labeled_skipped_char_flags * real_skipped_char_flags).view(-1).cpu().numpy())
				total_real_skipped_char_count += sum(real_skipped_char_flags.view(-1).cpu().numpy())

				# This takes into consideration how frequently false positives occur.
				total_correct_unfiltered_skipped_char_sequence_outputs += sum((correctly_labeled_skipped_char_flags * real_character_flags).view(-1).cpu().numpy())
			
				# print(torch.topk(predicted_skipped_char_labels, 1).indices)
				# print(predicted_skipped_char_locations[:10,0,:])

				skipped_char_location_errors = (predicted_skipped_char_locations - skipped_char_location_sequence_batch)
				skipped_char_location_error_distances = torch.sqrt(skipped_char_location_errors[:,:,0]**2 + skipped_char_location_errors[:,:,1]**2)
				total_distance_from_real_skipped_chars += sum((
					(skipped_char_location_error_distances * real_skipped_char_flags.view(skipped_char_location_error_distances.shape).float()).view(-1).cpu().numpy()))

				# RECORD THE LOSSES.
				testing_batch_classification_losses.append(classification_loss.detach().cpu().numpy())
				testing_batch_regression_losses.append(regression_loss.detach().cpu().numpy())
				testing_batch_skipped_char_classification_losses.append(skipped_char_classification_loss.detach().cpu().numpy())
				testing_batch_skipped_char_regression_losses.append(skipped_char_regression_loss.detach().cpu().numpy())

		# DISPLAY THE MODEL'S PERFORMANCE STATISTICS.
		# tr_c = training current character classification loss
		# tr_nr = training current character location regression loss
		# te_c = testing current character classification loss
		# te_nr = testing current character location regression loss
		# te_acc = testing current character classification accuracy
 		# te_n_mu_d = testing mean distance between character's predicted location and its actual location (normalized for image size).
		info_format = 'Epoch {}/{} -\ttr_c: {:.3f}, tr_r: {:.3f}, te_c: {:.3f}, te_r: {:.3f}, te_acc: {:.2f}%, te_mu_d: {:.2f}'
		print(info_format.format(
			epoch + 1, 
			EPOCH_COUNT,
			np.mean(training_batch_classification_losses),
			np.mean(training_batch_regression_losses),
			np.mean(testing_batch_classification_losses),
			np.mean(testing_batch_regression_losses),
			100*total_correct_character_label_count/total_real_character_count,
			total_distance_from_real_points/total_real_character_count))
		skipped_chars_info_format = '\t\ttr_sc: {:.3f}, tr_sr: {:.3f}, te_sc: {:.3f}, te_sr: {:.3f}, te_sacc: {:.2f}%, te_uf_sacc: {:.2f}%, te_mu_sd: {:.2f}'#, te_acc: {:.2f}%, te_mu_d: {:.2f}'
		print(skipped_chars_info_format.format(
			np.mean(training_batch_skipped_char_classification_losses),
			np.mean(training_batch_skipped_char_regression_losses),
			np.mean(testing_batch_skipped_char_classification_losses),
			np.mean(testing_batch_skipped_char_regression_losses),
			100*total_correct_skipped_char_label_count/total_real_skipped_char_count,
			100*total_correct_unfiltered_skipped_char_sequence_outputs/total_real_character_count, # total_real_character_count used as a proxy for the total length of the ground truth sequences.
			total_distance_from_real_skipped_chars/total_real_skipped_char_count))

		torch.save(model.state_dict(), 'TempModel.pth')
		