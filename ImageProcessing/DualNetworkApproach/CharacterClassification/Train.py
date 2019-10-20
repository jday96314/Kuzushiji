from apex import amp
import re
import os
import numpy as np
import pickle
import PIL
from PIL import ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, SequentialSampler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from DenseCharacterClassifier import *
from StackedFeatureExtractorAndCharacterClassifier import *

import sys
sys.path.append("../FeatureExtractors")
sys.path.append("../RegionProposal")
from CpuNonMaxSuppression import SuppressNonMaximalBboxes
from VggFeatureExtractor import VGG
from ResNetFeatureExtractor import *
from RPN_WithHidden import RPN_WithHidden
from StackedFeatureExtractorAndRpn import StackedFeatureExtractorAndRpn

SCALED_IMAGE_WIDTHS = 512
SCALED_IMAGE_HEIGHTS = 512
IMAGE_CHANNELS = 3
BATCH_SIZE = 4
GROUND_TRUTH_CSV_FILEPATH = '../../Datasets/scaled_train.csv'
DEVICE = torch.device("cuda")

def GetImageIdsFromFile(file_path):
	with open(file_path) as file:
		lines = file.read().split('\n')[1:]

	return [
		line.split(',')[0]
		for line in lines
		if len(line) > 0]

def GetImage(image_path):
	raw_image = PIL.Image.open(image_path)
	image = np.asarray(raw_image).T.reshape(
		IMAGE_CHANNELS,
		SCALED_IMAGE_WIDTHS,
		SCALED_IMAGE_HEIGHTS) / 255

	return image

def GetBatchRoisAndLabels(all_rois, all_roi_char_labels, batch_image_ids):
	batch_rois = []
	batch_roi_char_labels = []
	for new_id, old_id in enumerate(batch_image_ids):
		relevant_indices = torch.nonzero(all_rois[:, 0] == old_id).view(-1)
		image_rois = all_rois[relevant_indices]
		image_rois[:, 0] = new_id
		batch_rois.append(image_rois)

		image_labels = all_roi_char_labels[relevant_indices]
		batch_roi_char_labels.append(image_labels)

	return torch.cat(batch_rois), torch.cat(batch_roi_char_labels)

def DumpRpnOutputToFile(images, output_regions_filepath):
	# LOAD THE ANCHORS.
	anchors = pickle.load(open('../RegionProposal/SavedModels/anchors.p', 'rb'))

	# CREATE THE MODEL.
	feature_extractor = ResNet34(IMAGE_CHANNELS, filter_count_coef = 128, dropout_rate = 0)
	rpn_network = RPN_WithHidden(
		input_channels = feature_extractor.FinalChannelsCount, 
		anchor_count = len(anchors), 
		classifier_dropout_rate=.5,
		regression_dropout_rate=.5,
		classifier_hidden_units = 512, #256
		regressor_hidden_units = 512)

	model = StackedFeatureExtractorAndRpn(feature_extractor, rpn_network)
	model = model.to(DEVICE)
	model = torch.nn.DataParallel(model, device_ids=[0,1])

	# LOAD THE MODEL'S WEIGHTS.
	model.load_state_dict(torch.load(
		'../RegionProposal/SavedModels/StackedFeatureExtractorAndRpn_FullDataset_Color_DoubleWidthResNet34_100Epochs.pth'))

	# SET THE MODEL TO EVALUAITON MODE.
	model.eval()
	with torch.no_grad():
		# PROCESS ALL THE IMAGES.
		all_batches = list(BatchSampler(
			SequentialSampler(range(len(images))),
			batch_size=16, 
			drop_last=False))
		all_bboxes = []
		for batch_num in range(len(all_batches)):
			# GET THE MODEL'S OUTPUT FOR THE IMAGES.
			batch_indices = all_batches[batch_num]
			batch_images = images[batch_indices].to(DEVICE)
			predicted_region_class_labels, region_regression_results = model(batch_images)

			# GET THE MAXIMAL BBOXES.
			for batch_image_index in range(len(batch_indices)):
				maximal_bboxes = SuppressNonMaximalBboxes(
					anchors, 
					predicted_region_class_labels[batch_image_index], 
					region_regression_results[batch_image_index],
					SCALED_IMAGE_WIDTHS,
					SCALED_IMAGE_HEIGHTS,
					min_confidence_threshold = .5,
					min_suppression_ios = 1)
				all_bboxes += [
					[
						batch_indices[batch_image_index],
						bbox_center_x - bbox_w/2,
						bbox_center_y - bbox_h/2,
						bbox_center_x + bbox_w/2,
						bbox_center_y + bbox_h/2
					]
					for (bbox_center_x, bbox_center_y, bbox_w, bbox_h) in maximal_bboxes
				]

	pickle.dump(torch.tensor(all_bboxes), open(output_regions_filepath, 'wb'))

def GetBboxesBboxLabelsAndCharset(ground_truth_lines):
	all_images_bboxes = []
	all_images_bbox_unicode_labels = []
	all_unique_unicode_labels = set({})

	for image_data in ground_truth_lines:
		label_extraction_regex = bounding_box_parsing_regex = r'(U\+[A-Z0-9]*) ([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)'
		detected_bounding_boxes = re.findall(bounding_box_parsing_regex, image_data)
		image_bboxes = []
		image_bbox_unicode_labels = []
		for unicode_val, top_left_x, top_left_y, w, h in detected_bounding_boxes:
			image_bboxes.append((int(top_left_x), int(top_left_y), int(w), int(h)))
			image_bbox_unicode_labels.append(unicode_val)
			all_unique_unicode_labels.add(unicode_val)

		all_images_bboxes.append(image_bboxes)
		all_images_bbox_unicode_labels.append(image_bbox_unicode_labels)

	all_unique_unicode_labels = sorted(list(all_unique_unicode_labels))

	all_images_bbox_ord_labels = []
	for image_bbox_unicode_labels in all_images_bbox_unicode_labels:
		image_bbox_ord_labels = [all_unique_unicode_labels.index(unicode_label) for unicode_label in image_bbox_unicode_labels]
		all_images_bbox_ord_labels.append(image_bbox_ord_labels)

	return all_images_bboxes, all_images_bbox_ord_labels, all_unique_unicode_labels


def OpenImage(ground_truth_csv_filepath, images_directory, image_index):
	# 1 is added because the file contains a header row.
	image_description = open(ground_truth_csv_filepath).readlines()[image_index + 1]
	image_id, raw_bboxes = image_description.split(',')
	image_path = os.path.join(images_directory, image_id + '.png')
	raw_image = PIL.Image.open(image_path)
	return raw_image

def VisualizeWithBoxes(image, boxes, output_path):
	imsource = image.convert('RGBA')
	box_canvas = PIL.Image.new('RGBA', imsource.size)
	box_draw = ImageDraw.Draw(box_canvas)
	for _, top_left_x, top_left_y, bottom_right_x, bottom_right_y in boxes:
		box_draw.rectangle(
			(top_left_x, top_left_y, bottom_right_x, bottom_right_y),
			fill = (255, 255, 255, 0),
			outline = (255, 0, 0, 155))
	imsource = PIL.Image.alpha_composite(imsource, box_canvas)
	imsource = imsource.convert('RGB')
	imsource.save(output_path, 'PNG')


def TrainModel(
		initial_stacked_model_path = 'SavedModels/StackedFeatureExtractorAndCharacterClassifier_Color_FullTrainingSet.pth', 
		stacked_model_output_path = 'SavedModels/StackedFeatureExtractorAndCharacterClassifier.pth',
		cross_validate = True,
		recompute_char_class_labels = False,
		recompute_rpn_outputs = False):
	# LOAD THE TRAINING DATA.
	# all_image_ids = GetImageIdsFromFile(GROUND_TRUTH_CSV_FILEPATH)
	# image_paths = [
	# 	os.path.join('../../Datasets/Color/preprocessed_train_images', image_id + '.png')
	# 	for image_id in all_image_ids]
	# all_images = [GetImage(path) for path in image_paths]
	# all_images = torch.tensor(all_images, dtype = torch.float32)
	# pickle.dump(all_images, open('TrainingImages_Color.p', 'wb'), protocol = 4)
	all_images = pickle.load(open('TrainingImages_Color.p', 'rb'))

	# The rois are (center_x, center_y, w, h)
	RPN_OUTPUT_PATH = 'RpnNetworkOutputs/ResNet/IndependentlyTrainedRpnOutput.p'
	if recompute_rpn_outputs:
		DumpRpnOutputToFile(all_images, RPN_OUTPUT_PATH)
	all_rois = pickle.load(open(RPN_OUTPUT_PATH, 'rb'))

	# debug_image = OpenImage(
	# 	GROUND_TRUTH_CSV_FILEPATH, 
	# 	'../../Datasets/preprocessed_train_images',
	# 	0)
	# relevant_roi_indices = torch.nonzero((all_rois[:, 0] == 0)).view(-1)
	# relevant_rois = all_rois[relevant_roi_indices]	
	# VisualizeWithBoxes(debug_image, relevant_rois.numpy(), 'debug.png')
	# exit()

	# Dumping the ROI char class labels is only necessary when they can't be read from cache (i.e. when training stage 1 isn't being performed.)
	if recompute_char_class_labels:
		print('Recomputing the ROI char labels...')
		ground_truth_lines = [line for line in open(GROUND_TRUTH_CSV_FILEPATH).readlines()[1:]]
		# These bboxes are (top_left_x, top_left_y, w, h)
		all_images_bboxes, all_images_bbox_ord_labels, all_unique_unicode_labels = GetBboxesBboxLabelsAndCharset(ground_truth_lines)
		max_bbox_ord_label = max([max([-1] + image_labels) for image_labels in all_images_bbox_ord_labels]) # [-1] is present to avoid taking the max of empty lists.
		roi_char_class_labels = np.ones(len(all_rois)) * (max_bbox_ord_label + 1) # Labels that don't correspond to real chars are for ROIs whose centers aren't in a real bbox.
		for image_id, image_bboxes in enumerate(all_images_bboxes):
			print('Processing image {}/{}...'.format(image_id + 1, len(all_images)))
			relevant_roi_indices = torch.nonzero((all_rois[:, 0] == image_id)).view(-1)
			relevant_rois = all_rois[relevant_roi_indices]

			for current_image_roi_index in range(len(relevant_rois)):
				#_, roi_center_x, roi_center_y, roi_w, roi_h = relevant_rois[current_image_roi_index]
				_, roi_top_left_x, roi_top_left_y, roi_bottom_right_x, roi_bottom_right_y = relevant_rois[current_image_roi_index]
				roi_center_x = .5*(roi_top_left_x + roi_bottom_right_x)
				roi_center_y = .5*(roi_top_left_y + roi_bottom_right_y)
				for current_image_bbox_index in range(len(image_bboxes)):
					bbox_top_left_x, bbox_top_left_y, bbox_w, bbox_h = image_bboxes[current_image_bbox_index]
					roi_center_x_in_bbox = ((roi_center_x - bbox_top_left_x) < bbox_w) and (roi_center_x >= bbox_top_left_x)
					roi_center_y_in_bbox = ((roi_center_y - bbox_top_left_y) < bbox_h) and (roi_center_y >= bbox_top_left_y)
					roi_center_in_bbox = roi_center_x_in_bbox and roi_center_y_in_bbox
					if roi_center_in_bbox:
						roi_char_class_labels[relevant_roi_indices[current_image_roi_index]] = all_images_bbox_ord_labels[image_id][current_image_bbox_index]

		print('{}/{} rois are centered in a real bbox'.format(
			sum(roi_char_class_labels != max_bbox_ord_label + 1),
			len(roi_char_class_labels)))
		pickle.dump(roi_char_class_labels, open('RpnNetworkOutputs/ResNet/RoiCharClassLabels.p', 'wb'))

	all_roi_char_labels = torch.tensor(pickle.load(open('RpnNetworkOutputs/ResNet/RoiCharClassLabels.p', 'rb')), dtype=torch.long)

	all_rois = torch.index_select(all_rois, 1, torch.tensor([0,2,1,4,3], dtype = torch.long))

	# CREATE THE MODEL.
	feature_extractor = ResNet34(IMAGE_CHANNELS, filter_count_coef = 190, dropout_rate = .5)
	BATCH_SIZE = 2

	roi_pool_size = (2,2)
	character_classifier = DenseCharacterClassifier(
		input_feature_count = feature_extractor.FinalChannelsCount * roi_pool_size[0] * roi_pool_size[1], 
		hidden_sizes = [3072], #[2048],
		dropout_rates_after_hidden_layers = [.5],
		num_classes = max(all_roi_char_labels.numpy()) + 1,
		hidden_activation = 'ReLU')
	model = StackedFeatureExtractorAndCharacterClassifier(
		feature_extractor, 
		roi_pool_size, 
		character_classifier, 
		dropout_rate = .75,
		spatial_scale = 1/8) # TODO: This shouldn't be hardcoded.
	model = model.to(DEVICE)

	optimizer = optim.Adam(model.parameters(), .0002)
	learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(
		optimizer, 
		# milestones = [40, 50, 60, 70],
		milestones = [10, 20, 25],
		gamma = .5)
	LAST_EPOCH = 0
	for i in range(LAST_EPOCH): # For some reason last_epoch in the scheduler is busted.
		learning_rate_scheduler.step()

	# CONVERT THE MODEL AND OPTIMIZER TO MIXED PRECISION.
	model, optimizer = amp.initialize(
		model,
		optimizer,
		opt_level="O1",
		loss_scale="dynamic")

	# LOAD PRE-TRAINED WEIGHTS.
	if initial_stacked_model_path is not None:
		state_dict = torch.load(initial_stacked_model_path)
		del state_dict['CharacterClassifier.HiddenLayers.0.weight']
		del state_dict['CharacterClassifier.HiddenLayers.0.bias']
		del state_dict['CharacterClassifier.OutputLayer.weight']
		model.load_state_dict(
			state_dict,
			strict=False) # Possibly revert to True.

	# DETERMINE WHICH EXAMPLES ARE USED FOR TRAINING AND TESTING.
	if cross_validate:
		training_indices = np.array([i for i in range(len(all_images)) if i%4 != 0])
		testing_indices = np.array([i for i in range(len(all_images)) if i%4 == 0])	
		print('Using {} images for training and {} for testing.'.format(
			len(training_indices), 
			len(testing_indices)))
	else:
		training_indices = np.array(range(len(all_images)))
		print('Training on {} images.'.format(len(all_images)))

	# TRAIN THE MODEL.
	print('Training model...')
	# EPOCH_COUNT = 80
	EPOCH_COUNT = 30
	for epoch in range(LAST_EPOCH, EPOCH_COUNT):
		learning_rate_scheduler.step()
		print('Epoch {}/{}'.format(epoch + 1, EPOCH_COUNT))

		# TRAIN THE NETWORK.
		training_batch_losses = []
		training_batch_f1_scores = []
		training_batch_accuracies = []
		all_training_batches = list(BatchSampler(
			SequentialSampler(training_indices), # Needs to be sequential because of how the batch ROIs are retrieved.
			batch_size=BATCH_SIZE, 
			drop_last=False))
		for batch_num in range(len(all_training_batches)):
			# SET THE MODEL TO TRAINING MODE.
			model.train()

			# GET THE BATCH DATA.
			batch_indices = training_indices[all_training_batches[batch_num]]
			batch_images = all_images[batch_indices]
			batch_images = batch_images.to(DEVICE)
			batch_rois, batch_real_char_labels = GetBatchRoisAndLabels(all_rois, all_roi_char_labels, batch_indices)
			if (batch_rois.shape[0] > 100*BATCH_SIZE):
				# There may be multiple bounding boxes per character, so a random subset is
				# chosen (without replacement).
				subset_indices = torch.randperm(batch_rois.shape[0])[:150*BATCH_SIZE]
				batch_rois = batch_rois[subset_indices]
				batch_real_char_labels = batch_real_char_labels[subset_indices]
			if batch_rois.shape[0] == 0:
				continue

			batch_rois = batch_rois.to(DEVICE)
			batch_real_char_labels = batch_real_char_labels.to(DEVICE)

			# ZERO THE GRADIENTS.
			model.zero_grad()

			# FORWARD PASS.
			predicted_chars = model(batch_images, batch_rois.half())

			# COMPUTE LOSSES.
			loss_function = nn.CrossEntropyLoss().to(DEVICE)
			loss = loss_function(
				predicted_chars,
				batch_real_char_labels)

			# UPDATE THE NETWORK.
			with amp.scale_loss(loss, optimizer) as scale_loss: # amp
				scale_loss.backward()
			optimizer.step()

			# SAVE THE LOSS.
			training_batch_losses.append(loss.detach().cpu().numpy())

			# COMPUTE THE HUMAN READABLE METRICS.
			# For performance reasons, this is only done for 20% of the data. It takes 
			# roughly .03 seconds to process 32 images. This adds up over many epochs.
			if batch_num%5 == 0:
				batch_predicted_chars_ords = np.argmax(predicted_chars.detach().cpu().numpy(), axis = 1)
				batch_real_char_labels = batch_real_char_labels.detach().cpu().numpy()
				training_batch_f1_scores.append(f1_score(batch_real_char_labels, batch_predicted_chars_ords, average = 'weighted'))
				training_batch_accuracies.append(accuracy_score(batch_real_char_labels, batch_predicted_chars_ords))

		if cross_validate:
			# TEST THE NETWORK.
			model.eval()
			with torch.no_grad():
				testing_batch_losses = []
				testing_batch_f1_scores = []
				testing_batch_accuracies = []
				all_testing_batches = list(BatchSampler(
					SequentialSampler(testing_indices), # Needs to be sequential because of how the batch ROIs are retrieved.
					batch_size=BATCH_SIZE * 2, # Not having gradients allows you to get away with a larger batch size. 
					drop_last=False))
				for batch_num in range(len(all_testing_batches)):
					# GET THE BATCH DATA.
					batch_indices = testing_indices[all_testing_batches[batch_num]]
					batch_images = all_images[batch_indices].to(DEVICE)
					batch_rois, batch_real_char_labels = GetBatchRoisAndLabels(all_rois, all_roi_char_labels, batch_indices)
					if (batch_rois.shape[0] > 150*BATCH_SIZE):
						# There may be multiple bounding boxes per character, so a random subset is
						# chosen (without replacement).
						subset_indices = torch.randperm(batch_rois.shape[0])[:150*BATCH_SIZE]
						batch_rois = batch_rois[subset_indices]
						batch_real_char_labels = batch_real_char_labels[subset_indices]
					if batch_rois.shape[0] == 0:
						continue
					batch_rois = batch_rois.to(DEVICE)
					batch_real_char_labels = batch_real_char_labels.to(DEVICE)

					# FORWARD PASS.
					predicted_chars = model(batch_images, batch_rois.half())

					# COMPUTE LOSSES.
					loss_function = nn.CrossEntropyLoss().to(DEVICE)
					loss = loss_function(
						predicted_chars,
						batch_real_char_labels)

					# SAVE THE LOSS.
					testing_batch_losses.append(loss.detach().cpu().numpy())

					# COMPUTE THE HUMAN READABLE METRICS.
					batch_predicted_chars_ords = np.argmax(predicted_chars.detach().cpu().numpy(), axis = 1)
					batch_real_char_labels = batch_real_char_labels.detach().cpu().numpy()
					testing_batch_f1_scores.append(f1_score(batch_real_char_labels, batch_predicted_chars_ords, average = 'weighted'))
					testing_batch_accuracies.append(accuracy_score(batch_real_char_labels, batch_predicted_chars_ords))

		if cross_validate:
			print('\tTesting - Loss mu: {:.2f}, f1: {:.4f}, Accuracy: {:.4f}'.format(
				np.mean(testing_batch_losses),
				np.mean(testing_batch_f1_scores),
				np.mean(testing_batch_accuracies)))
		print('\tTraining - Loss mu: {:.2f}, f1: {:.4f}, Accuracy: {:.4f}'.format(
			np.mean(training_batch_losses),
			np.mean(training_batch_f1_scores),
			np.mean(training_batch_accuracies)))

		# SAVE THE MODEL.
		torch.save(model.state_dict(), stacked_model_output_path)
	
if __name__ == '__main__':
	# TrainModel(
	# 	recompute_char_class_labels = False,
	# 	recompute_rpn_outputs = False)

	TrainModel(
		recompute_char_class_labels = False,
		recompute_rpn_outputs = False,
		cross_validate = False)
