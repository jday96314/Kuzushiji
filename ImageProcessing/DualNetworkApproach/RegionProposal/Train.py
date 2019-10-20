from apex import amp
import PIL
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.optim as optim
import math
import re
import os
import numpy as np
import pickle
import torchvision
from torch.utils.data import BatchSampler, RandomSampler

import sys
sys.path.append("../FeatureExtractors")

from RPN import *
from RPN_WithHidden import *
from CpuNonMaxSuppression import *
from StackedFeatureExtractorAndRpn import *
from VggFeatureExtractor import VGG
from ResNetFeatureExtractor import *

SCALED_IMAGE_WIDTHS = 512
SCALED_IMAGE_HEIGHTS = 512
IMAGE_CHANNELS = 3
USE_PRECOMPUTED_ANCHORS = True
USE_PRECOMPUTED_TRAINING_DATA = True
ANCHORS_FILEPATH = 'SavedModels/anchors.p'
TRAINING_DATA_FILEPATH = 'training_data_modified_test.p'
GROUND_TRUTH_CSV_FILEPATH = '../../Datasets/scaled_train.csv'
TRAINING_DATA_DIRECTORY = '../../Datasets/Color/preprocessed_train_images'
DEVICE = torch.device("cuda")
FMAP_WIDTH, FMAP_HEIGHT = 64, 64

def ComputeIntersectOverUnion(center_x_1, center_y_1, w1, h1, center_x_2, center_y_2, w2, h2):
	if abs(center_x_1 - center_x_2) > (w1//2 + w2//2):
		return 0
	if abs(center_y_1 - center_y_2) > (h1//2 + h2//2):
		return 0

	x1 = max(center_x_1 - w1/2, center_x_2 - w2/2)
	y1 = max(center_y_1 - h1/2, center_y_2 - h2/2)
	x2 = min(center_x_1 + w1/2, center_x_2 + w2/2)
	y2 = min(center_y_1 + h1/2, center_y_2 + h2/2)

	shared_area = max((x2 - x1) * (y2 - y1), 0)
	total_area = (w1*h1 + w2*h2) - shared_area

	return shared_area / total_area

# Label 1 if IOU > 0.7. Label 0 otherwise.
# Second output contains 0 at the indices that should not contribute to loss.
def GetAllAnchorClassLabelsAndAdjustmentsForImage(bboxes, anchors, feature_map_width, feature_map_height):
	class_labels = np.zeros(shape = (len(anchors), feature_map_width, feature_map_height))
	loss_mask = np.ones(shape = (2, len(anchors), feature_map_width, feature_map_height))
	bbox_adjustments = np.zeros(shape = (4, len(anchors), feature_map_width, feature_map_height))
	min_anchor_distances = 1e9 * np.ones(shape = (len(anchors), feature_map_width, feature_map_height))

	for bbox_x, bbox_y, bbox_w, bbox_h in bboxes:
		bbox_center_x = bbox_x + bbox_w/2 # The raw xy is in the top left.
		bbox_center_y = bbox_y + bbox_h/2

		# These are rounded down indices of the feature vectors which correspond to the center
		# of the bounding boxes.
		lower_bound_fmap_x_index = int(bbox_center_x * (feature_map_width / SCALED_IMAGE_WIDTHS))
		lower_bound_fmap_y_index = int(bbox_center_y * (feature_map_height / SCALED_IMAGE_HEIGHTS))

		max_gt_bbox_iou = -1 # The max amount of overlap between the current ground truth bounding box and any anchor.
		max_iou_fmap_x_index = 0
		max_iou_fmap_y_index = 0 
		max_iou_anchor_index = 0
		debug_max_iou_anchor_location = None

		for anchor_index in range(len(anchors)):
			anchor_w, anchor_h = anchors[anchor_index]

			if lower_bound_fmap_y_index < feature_map_height//4:
				min_y_offset = -lower_bound_fmap_y_index
				max_y_offset = feature_map_height//2 - lower_bound_fmap_y_index
			elif lower_bound_fmap_y_index < 3*feature_map_height//4:
				# min_y_offset = -feature_map_height//4
				# max_y_offset = feature_map_height//4
				min_y_offset = -lower_bound_fmap_y_index
				max_y_offset = feature_map_height - lower_bound_fmap_y_index - 1
			else:
				min_y_offset = -feature_map_height//2 + (feature_map_height - lower_bound_fmap_y_index)
				max_y_offset = feature_map_height - lower_bound_fmap_y_index - 1

			if lower_bound_fmap_x_index < feature_map_width//4:
				min_x_offset = -lower_bound_fmap_x_index
				max_x_offset = feature_map_width//2 - lower_bound_fmap_x_index
			elif lower_bound_fmap_x_index < 3*feature_map_width//4:
				min_x_offset = -lower_bound_fmap_x_index
				max_x_offset = feature_map_width - lower_bound_fmap_x_index - 1
			else:
				min_x_offset = -feature_map_width//2 + (feature_map_width - lower_bound_fmap_x_index)
				max_x_offset = feature_map_width - lower_bound_fmap_x_index - 1

			for fmap_y_offset in range(min_y_offset, max_y_offset + 1):
				for fmap_x_offset in range(min_x_offset, max_x_offset + 1):
					# VALIDATE THE OFFSETS.
					# This is to guard against bugs -- the warnings should never be visible.
					fmap_x_index = lower_bound_fmap_x_index + fmap_x_offset
					if fmap_x_index >= feature_map_width or fmap_x_index < 0:
						print('Skipping invalid fmap x index.', lower_bound_fmap_x_index, fmap_x_offset)
						continue
					
					fmap_y_index = lower_bound_fmap_y_index + fmap_y_offset
					if fmap_y_index >= feature_map_height or fmap_y_index < 0:
						print('Skipping invalid fmap y index.', lower_bound_fmap_y_index, fmap_y_offset)
						continue

					# COMPUTE THE SIZE OF THE INTERSECTION BETWEEN THE ANCHOR AND THE BBOX.
					anchor_x = (fmap_x_index + .5) * (SCALED_IMAGE_WIDTHS / feature_map_width) # The .5 is necessary so that this will be the anchor center. Please test this.
					anchor_y = (fmap_y_index + .5) * (SCALED_IMAGE_HEIGHTS / feature_map_height)

					iou = ComputeIntersectOverUnion(
						bbox_center_x, bbox_center_y, bbox_w, bbox_h,
						anchor_x, anchor_y, anchor_w, anchor_h)

					# UPDATE THE ANCHOR'S CLASS LABEL & LABEL MASK.
					# For performance reasons, we first check if an update is likely to be necessary.
					if fmap_x_offset in [0,1] and fmap_y_offset in [0,1]:
						if iou > .7:
							class_labels[anchor_index][fmap_x_index][fmap_y_index] = 1
							loss_mask[0][anchor_index][fmap_x_index][fmap_y_index] = 1
							loss_mask[1][anchor_index][fmap_x_index][fmap_y_index] = 1

						elif iou  > .3 and class_labels[anchor_index][fmap_x_index][fmap_y_index] == 0:
							loss_mask[0][anchor_index][fmap_x_index][fmap_y_index] = 0
							loss_mask[1][anchor_index][fmap_x_index][fmap_y_index] = 0

						if iou > max_gt_bbox_iou:
							# Used for updating labels later.
							max_iou_fmap_x_index = fmap_x_index
							max_iou_fmap_y_index = fmap_y_index
							max_iou_anchor_index = anchor_index
							max_gt_bbox_iou = iou

					# SET THE ANCHOR ADJUSTMENT REGRESSION TARGETS.
					# See https://arxiv.org/pdf/1506.01497.pdf for explanation of targets (t*).
					distance_between_anchor_and_bbox = ((bbox_center_x - anchor_x)**2 + (bbox_center_y - anchor_y)**2)**.5
					if distance_between_anchor_and_bbox < min_anchor_distances[anchor_index][fmap_x_index][fmap_y_index]:
						tx = (bbox_center_x - anchor_x)/anchor_w
						ty = (bbox_center_y - anchor_y)/anchor_h
						tw = math.log(bbox_w/anchor_w)
						th = math.log(bbox_h/anchor_h)
						bbox_adjustments[0][anchor_index][fmap_x_index][fmap_y_index] = tx
						bbox_adjustments[1][anchor_index][fmap_x_index][fmap_y_index] = ty
						bbox_adjustments[2][anchor_index][fmap_x_index][fmap_y_index] = tw
						bbox_adjustments[3][anchor_index][fmap_x_index][fmap_y_index] = th

						min_anchor_distances[anchor_index][fmap_x_index][fmap_y_index] = distance_between_anchor_and_bbox

		# The best matching anchor should always be in the overlapping class, even if it doesn't
		# meet the threshold.
		class_labels[max_iou_anchor_index][max_iou_fmap_x_index][max_iou_fmap_y_index] = 1
		loss_mask[0][max_iou_anchor_index][max_iou_fmap_x_index][max_iou_fmap_y_index] = 1
		loss_mask[1][max_iou_anchor_index][max_iou_fmap_x_index][max_iou_fmap_y_index] = 1

	return class_labels, loss_mask, bbox_adjustments

def GetRpnTrainingDataForGroundTruthLine(args):
	line, anchors = args
	image_id, raw_bboxes = line.split(',')

	# PARSE THE BOUNDING BOXES.
	bounding_box_parsing_regex = r'(U\+[A-Z0-9]*) ([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)'
	detected_bounding_boxes = re.findall(bounding_box_parsing_regex, raw_bboxes)
	bboxes = []
	# x & y are in the top left.
	for unicode_val, x, y, w, h in detected_bounding_boxes:
		# X & Y are in the top left
		bboxes.append((int(x), int(y), int(w), int(h)))

	# PARSE THE IMAGE.
	try:
		image_path = os.path.join(TRAINING_DATA_DIRECTORY, image_id + '.png')
		raw_image = PIL.Image.open(image_path)
	except:
		image_path = os.path.join(FALLBACK_TRAINING_DATA_DIRECTORY, image_id + '.png')
		raw_image = PIL.Image.open(image_path)
	image = np.asarray(raw_image).T.reshape( # The transpose is necessary because the default shape is (h, w)
		IMAGE_CHANNELS,
		SCALED_IMAGE_WIDTHS,
		SCALED_IMAGE_HEIGHTS) / 255

	# GET THE GROUND TRUTH CLASS LABELS FOR THE ANCHORS.
	class_labels, class_loss_mask, bbox_adjustments = GetAllAnchorClassLabelsAndAdjustmentsForImage(
		bboxes, anchors, FMAP_WIDTH, FMAP_HEIGHT)

	return image, class_labels, class_loss_mask, bbox_adjustments

def TrainModel(
		initial_stacked_model_path = None, 
		stacked_model_output_path = 'SavedModels/StackedFeatureExtractorAndRpn.pth',
		cross_validate = True):
	# GET THE ANCHOR SIZES.
	print('Loading anchors...')
	if USE_PRECOMPUTED_ANCHORS:
		anchors = pickle.load(open(ANCHORS_FILEPATH, 'rb'))
	else:
		anchors = ComputeAnchorSizes(GROUND_TRUTH_CSV_FILEPATH)
		pickle.dump(anchors, open(ANCHORS_FILEPATH, 'wb'))

	# LOAD THE DATASET.
	print('Loading training data...')
	if USE_PRECOMPUTED_TRAINING_DATA:
		all_images, all_anchor_class_labels, all_anchor_class_label_loss_masks, all_anchor_regression_targets = pickle.load(open(TRAINING_DATA_FILEPATH, 'rb'))
	else:
		worker_args = [(line, anchors) for line in open(GROUND_TRUTH_CSV_FILEPATH).readlines()[1:]]
		with Pool(7) as worker_pool:
			worker_results = worker_pool.map(GetRpnTrainingDataForGroundTruthLine, worker_args)

		all_images = []
		all_anchor_class_labels = []
		all_anchor_class_label_loss_masks = []
		all_anchor_regression_targets = []

		for image, class_labels, class_loss_mask, bbox_adjustments in worker_results:
			all_images.append(image)
			all_anchor_class_labels.append(class_labels)
			all_anchor_class_label_loss_masks.append(class_loss_mask)
			all_anchor_regression_targets.append(bbox_adjustments)
			
		all_images = torch.tensor(all_images, dtype = torch.float32)
		all_anchor_class_labels = torch.tensor(all_anchor_class_labels, dtype = torch.long)
		all_anchor_class_label_loss_masks = torch.tensor(all_anchor_class_label_loss_masks, dtype = torch.float32)
		all_anchor_regression_targets = torch.tensor(all_anchor_regression_targets, dtype=torch.float32)
		pickle.dump(
			(all_images, all_anchor_class_labels, all_anchor_class_label_loss_masks, all_anchor_regression_targets), 
			open(TRAINING_DATA_FILEPATH, 'wb'),
			protocol = 4)

	# CREATE THE MODEL.
	print('Creating model...')

	# Thoughts thus far...
	# - Resnets seem to work better than VGG.
	# - ResNet50 with a filter count coef of 64 seems to overfit. It works better with a
	#   filter count coef of 32. The score for 64 you see below most likely would not have
	#   improved with additional epochs, but the score for 32 would.
	# - ResNet34 uses way less VRAM than ResNet50 (i.e. 7.7 GB with a batch size of 8 vs 6.4 with 2).
	# - ResNet34 seems to work better than ResNet50 (better loss)
	# - ResNet18 is inferior to 34 (at stock widths)
	# - ResNet34 works really well at 2x width. Dropout might be benificial
	#	because the test regression loss was much higher than the train regression loss.
	# - Instance norm resulted in slower training & better stability than batch norm. 
	# - Using a slight dropout just before regression input *might* be slightly benificial
	#	I would need to do more than 10 epochs to be sure.
	# - ResNet18 with a channel coef of 256 is inferior to 24 with a coef of 128
	
	feature_extractor = ResNet34(IMAGE_CHANNELS, filter_count_coef = 128, dropout_rate = .5)
	# feature_extractor = ResNet(
	# 	BasicBlock, 
	# 	[3,6,36,3], 
	# 	image_channels = 1, 
	# 	filter_count_coef = 128, 
	# 	dropout_rate = .4)

	# rpn_network = RPN(
	# 	input_channels = feature_extractor.FinalChannelsCount, 
	# 	anchor_count = len(anchors))
	rpn_network = RPN_WithHidden(
		input_channels = feature_extractor.FinalChannelsCount, 
		anchor_count = len(anchors), 
		classifier_dropout_rate=.5,
		regression_dropout_rate=.5,
		classifier_hidden_units = 512, #256
		regressor_hidden_units = 512) #256

	model = StackedFeatureExtractorAndRpn(feature_extractor, rpn_network)
	model = model.to(DEVICE)
	optimizer = optim.SGD(model.parameters(), .05, momentum=.9, nesterov=True)

	# CONVERT THE MODEL AND OPTIMIZER TO MIXED PRECISION.
	model, optimizer = amp.initialize(
		model,
		optimizer,
		opt_level="O1",
		loss_scale="dynamic")
	model = nn.DataParallel(model, device_ids=[0,1])

	# LOAD PRE-TRAINED WEIGHTS.
	if initial_stacked_model_path is not None:
		print('Loading pre-trained stacked model weights.')
		model.load_state_dict(
			torch.load(initial_stacked_model_path))

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
	EPOCH_COUNT = 100
	learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(
		optimizer, 
		milestones = [15, 50, 75, 90, 95],
		gamma = .5)
	for epoch in range(EPOCH_COUNT):
		print('Epoch {}/{}'.format(epoch + 1, EPOCH_COUNT))

		# TRAIN THE NETWORK.
		epoch_batch_training_classification_losses = []
		epoch_batch_training_regression_losses = []
		all_training_batches = list(BatchSampler(
			RandomSampler(training_indices),
			batch_size=4, # 4 for double width resnet 34
			drop_last=False))
		for batch_num in range(len(all_training_batches)):
			# SET THE MODEL TO TRAINING MODE.
			model.train()

			# GET THE BATCH DATA.
			batch_indices = training_indices[all_training_batches[batch_num]]
			batch_images = all_images[batch_indices].to(DEVICE)
			batch_anchor_classes = all_anchor_class_labels[batch_indices].to(DEVICE)
			batch_anchor_classes_loss_masks = all_anchor_class_label_loss_masks[batch_indices].to(DEVICE)
			batch_anchor_regression_targets = all_anchor_regression_targets[batch_indices].to(DEVICE)

			# ZERO THE GRADIENTS.
			model.zero_grad()

			# FORWARD PASS.
			predicted_region_class_labels, region_regression_results = model(batch_images)

			# COMPUTE LOSSES.
			classification_loss_function = nn.CrossEntropyLoss(
				weight = torch.tensor([1,15], dtype = torch.float32).to(DEVICE))
			classification_loss = classification_loss_function(
				predicted_region_class_labels * batch_anchor_classes_loss_masks,
				batch_anchor_classes)

			element_wise_regression_loss_function = nn.SmoothL1Loss(reduction = 'none')
			element_wise_regression_loss = element_wise_regression_loss_function(
				region_regression_results,
				batch_anchor_regression_targets)
			element_wise_regression_loss = torch.sum(element_wise_regression_loss, dim = 1, keepdim = True)
			element_wise_weights = batch_anchor_classes.float().view(element_wise_regression_loss.shape)
			regression_loss = 400 * torch.mean(element_wise_regression_loss * element_wise_weights)
			
			loss = classification_loss + regression_loss

			# UPDATE THE NETWORK.
			with amp.scale_loss(loss, optimizer) as scale_loss: # amp
				scale_loss.backward()
			optimizer.step()

			# SAVE THE LOSS.
			epoch_batch_training_classification_losses.append(classification_loss.detach().cpu().numpy())
			epoch_batch_training_regression_losses.append(regression_loss.detach().cpu().numpy())

		learning_rate_scheduler.step()

		if cross_validate:
			# SET THE MODEL TO EVALUATION MODE.
			model.eval()
			with torch.no_grad():
				# CROSS-VALIDATE THE NETWORK.
				epoch_batch_testing_classification_losses = []
				epoch_batch_testing_regression_losses = []
				all_testing_batches = list(BatchSampler(
					RandomSampler(testing_indices),
					batch_size=8,
					drop_last=False))
				for batch_num in range(len(all_testing_batches)):
					# GET THE BATCH DATA.
					batch_indices = testing_indices[all_testing_batches[batch_num]]
					batch_images = all_images[batch_indices].to(DEVICE)
					batch_anchor_classes = all_anchor_class_labels[batch_indices].to(DEVICE)
					batch_anchor_classes_loss_masks = all_anchor_class_label_loss_masks[batch_indices].to(DEVICE)
					batch_anchor_regression_targets = all_anchor_regression_targets[batch_indices].to(DEVICE)
			
					# FORWARD PASS.
					predicted_region_class_labels, region_regression_results = model(batch_images)

					# COMPUTE LOSSES.
					classification_loss_function = nn.CrossEntropyLoss(
						weight = torch.tensor([1,1], dtype = torch.float32).to(DEVICE))
					classification_loss = classification_loss_function(
						predicted_region_class_labels * batch_anchor_classes_loss_masks,
						batch_anchor_classes)

					element_wise_regression_loss_function = nn.SmoothL1Loss(reduction = 'none')
					element_wise_regression_loss = element_wise_regression_loss_function(
						region_regression_results,
						batch_anchor_regression_targets)
					element_wise_regression_loss = torch.sum(element_wise_regression_loss, dim = 1, keepdim = True)
					element_wise_weights = batch_anchor_classes.float().view(element_wise_regression_loss.shape)
					regression_loss = 400 * torch.mean(element_wise_regression_loss * element_wise_weights)
					
					loss = classification_loss + regression_loss

					# SAVE THE LOSS.
					epoch_batch_testing_classification_losses.append(classification_loss.detach().cpu().numpy())
					epoch_batch_testing_regression_losses.append(regression_loss.detach().cpu().numpy())

					# SAVE THE TRAINED MODEL.
					if stacked_model_output_path is not None:
						torch.save(model.state_dict(), stacked_model_output_path)

		if cross_validate:
			print('\tTesting mean loss -  c: {:.04f}, r: {:.04f}'.format(
				np.mean(epoch_batch_testing_classification_losses),
				np.mean(epoch_batch_testing_regression_losses)))
		print('\tTraining mean loss -  c: {:.04f}, r: {:.04f}'.format(
			np.mean(epoch_batch_training_classification_losses),
			np.mean(epoch_batch_training_regression_losses)))

		# SAVE THE TRAINED MODEL.
		if stacked_model_output_path is not None:
			torch.save(model.state_dict(), stacked_model_output_path)

if __name__ == '__main__':
	TrainModel(
		initial_stacked_model_path = None, #'SavedModels/StackedFeatureExtractorAndRpn_HugeResNetWithDropout_32Epochs_c059_r029.pth', 
		stacked_model_output_path = 'SavedModels/StackedFeatureExtractorAndRpn.pth',
		cross_validate = False)