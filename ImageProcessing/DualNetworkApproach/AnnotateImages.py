import os
import PIL
from PIL import Image
import apex
import numpy as np
import torch
import pickle
from torch.utils.data import BatchSampler, SequentialSampler
import math
import torch.nn.functional as F

from RegionProposal.StackedFeatureExtractorAndRpn import StackedFeatureExtractorAndRpn
from RegionProposal.RPN import RPN
from RegionProposal.RPN_WithHidden import RPN_WithHidden
from RegionProposal.CpuNonMaxSuppression import SuppressNonMaximalBboxes
from CharacterClassification.DenseCharacterClassifier import DenseCharacterClassifier
from CharacterClassification.StackedFeatureExtractorAndCharacterClassifier import StackedFeatureExtractorAndCharacterClassifier
from FeatureExtractors.VggFeatureExtractor import VGG
from FeatureExtractors.ResNetFeatureExtractor import *

RUN_ON_CROSS_VALIDATION_IMAGES = False
RUN_ON_FINAL_TEST_IMAGES = True

SCALED_IMAGE_WIDTHS = 512
SCALED_IMAGE_HEIGHTS = 512

DEVICE = torch.device("cuda")
MIXED_PRECISION_BATCH_SIZE = 32
FULL_PRECISION_BATCH_SIZE = 16

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
		-1,
		SCALED_IMAGE_WIDTHS,
		SCALED_IMAGE_HEIGHTS) / 255

	return image

def GetRois(images):
	# LOAD THE ANCHORS.
	anchors = pickle.load(open('RegionProposal/SavedModels/anchors.p', 'rb'))

	# CREATE THE MODEL.
	feature_extractor = ResNet34(image_channels=3, filter_count_coef = 128, dropout_rate = 0)
	rpn_network = RPN_WithHidden(
		input_channels = feature_extractor.FinalChannelsCount, 
		anchor_count = len(anchors), 
		classifier_dropout_rate = .5,
		regression_dropout_rate = .5,
		classifier_hidden_units = 256, #512, 
		regressor_hidden_units = 256) #512)

	model = StackedFeatureExtractorAndRpn(feature_extractor, rpn_network)
	model = model.to(DEVICE)
	model = torch.nn.DataParallel(model, device_ids=[0,1])

	# LOAD THE MODEL'S WEIGHTS.
	model.load_state_dict(torch.load(
		# 'RegionProposal/SavedModels/StackedFeatureExtractorAndRpn_FullDataset_Color_DoubleWidthResNet34_100Epochs.pth'))
		'RegionProposal/SavedModels/StackedFeatureExtractorAndRpn_DoubleWidthResNet34_Color_DropoutBefore1x1_DropoutInBlock_0417_017.pth'))

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
					min_confidence_threshold = .6, #.65,
					min_suppression_ios = .45) #.45)
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

	return torch.tensor(all_bboxes)

# The vals in batch_image_ids are NOT the same as the IDs in the image file names.
# They are expected to be numeric indices.
def GetBatchRois(all_rois, batch_image_ids):
	batch_rois = []
	for new_id, old_id in enumerate(batch_image_ids):
		relevant_indices = torch.nonzero(all_rois[:, 0] == old_id).view(-1)
		image_rois = all_rois[relevant_indices]
		image_rois[:, 0] = new_id
		batch_rois.append(image_rois)

	return torch.cat(batch_rois)

def GetUnicodeLabels(all_images, raw_rois):
	# The image x & y dims are expected to be the opposite of what the ROI pooling expects, so the 
	# bbox x&y coords also need to be swapped.
	all_rois = torch.index_select(raw_rois, 1, torch.tensor([0,2,1,4,3], dtype = torch.long))

	# LOAD THE CHARACTER SET.
	unique_unicode_vals = pickle.load(open('CharacterClassification/SavedModels/SortedCharset.p', 'rb'))

	# CREATE THE MODEL.
	IMAGE_CHANNELS = 1
	feature_extractor = feature_extractor = ResNet34(IMAGE_CHANNELS, filter_count_coef = 190, dropout_rate = .5)
	roi_pool_size = (2,2)
	character_classifier = DenseCharacterClassifier(
		input_feature_count = feature_extractor.FinalChannelsCount * roi_pool_size[0] * roi_pool_size[1], 
		hidden_sizes = [2048],
		dropout_rates_after_hidden_layers = [0],
		num_classes = len(unique_unicode_vals) + 1)
	model = StackedFeatureExtractorAndCharacterClassifier(
		feature_extractor, 
		roi_pool_size, 
		character_classifier,
		spatial_scale = 1/8) # TODO: This shouldn't be hardcoded.
	model = model.to(DEVICE)

	# LOAD THE TRAINED WEIGHTS.
	model.load_state_dict(torch.load(
		'CharacterClassification/SavedModels/StackedFeatureExtractorAndCharacterClassifier_FullTrainingSet.pth'))
		# 'CharacterClassification/SavedModels/StackedFeatureExtractorAndCharacterClassifier_ExtraWideResNet34_UberHeavyDropout_93test.pth'))

	# SET THE MODEL TO EVALUAITON MODE.
	model.eval()
	with torch.no_grad():
		# PROCESS ALL THE IMAGES.
		all_batches = list(BatchSampler(
			SequentialSampler(range(len(all_images))),
			batch_size=8, 
			drop_last=False))
		all_roi_chars = []
		all_confidences = []
		for batch_num in range(len(all_batches)):
			# LOAD THE BATCH DATA.
			batch_indices = np.array(all_batches[batch_num])
			batch_images = all_images[batch_indices].to(DEVICE)
			batch_rois = GetBatchRois(all_rois, batch_indices).to(DEVICE)

			# GET THE PREDICTED LABELS.
			raw_predicted_chars = F.softmax(model(batch_images, batch_rois), dim = -1).detach().cpu().numpy()
			for prediction in raw_predicted_chars:
				char_ord = np.argmax(prediction)
				confidence = prediction.max()
				all_confidences.append(confidence)
				if char_ord < len(unique_unicode_vals):
					all_roi_chars.append(unique_unicode_vals[char_ord])
				else:
					print('NA encountered', max(prediction))
					all_roi_chars.append('NA')

	return np.array(all_roi_chars), np.array(all_confidences)


# If include confidence is true then probs.max() will be included in the output
# instead of supressing oes which fall below a certain threshold.
def GenerateSubmissionFile(image_ids, greyscale_image_paths, color_image_paths, unscaled_images_dir, output_path, include_confidence = False):
	# LOAD THE IMAGES.
	print('Loading images...')
	greyscale_images = [GetImage(path) for path in greyscale_image_paths]
	greyscale_images = torch.tensor(greyscale_images, dtype = torch.float32)
	color_images = [GetImage(path) for path in color_image_paths]
	color_images = torch.tensor(color_images, dtype = torch.float32)

	# EXECUTE THE MODELS.
	print('Running models...')
	rois_tensor = GetRois(color_images)
	unicode_labels, label_confidences = GetUnicodeLabels(greyscale_images, rois_tensor)

	# WRITE THE OUTPUT FILE.
	print('Creating the output file...')
	image_ids_to_labels = {image_id:[] for image_id in image_ids}
	rois = rois_tensor.numpy()
	for roi_index in range(len(rois)):
		# LOAD THE CHARACTER TYPE.
		label = unicode_labels[roi_index]
		if label == 'NA':
			continue

		confidence = label_confidences[roi_index]
		if (not include_confidence) and (confidence < .75):
			continue 

		# DETERMINE THE IMAGE'S ID.
		image_id = image_ids[int(rois[roi_index][0])]

		# GET THE ROI CENTER.
		unsclaed_width, unscaled_height = PIL.Image.open(os.path.join(
			unscaled_images_dir, image_id + '.jpg')).size
		scaled_center_x = (rois[roi_index][1] + rois[roi_index][3]) / 2
		center_x = int(scaled_center_x * unsclaed_width / SCALED_IMAGE_WIDTHS)
		scaled_center_y = (rois[roi_index][2] + rois[roi_index][4]) / 2
		center_y = int(scaled_center_y * unscaled_height / SCALED_IMAGE_HEIGHTS)

		# STORE A STRING REPRESENTATION OF THE LABEL.
		if not include_confidence:
			image_label_str = '{} {} {}'.format(label, center_x, center_y)
		else:
			image_label_str = '{} {} {} {}'.format(label, confidence, center_x, center_y)
		#image_label_str = '{} {} {}'.format(label, center_y, center_x)
		image_ids_to_labels[image_id].append(image_label_str)

	with open(output_path, 'w') as output_file:
		output_file.write('image_id,labels\n')
		for image_id in image_ids_to_labels.keys():
			output_file.write('{},{}\n'.format(image_id, ' '.join(image_ids_to_labels[image_id])))

if __name__ == '__main__':
	# GENERATE THE SUBMISSION FILE(S).
	if RUN_ON_CROSS_VALIDATION_IMAGES:
		all_image_ids = GetImageIdsFromFile('../Datasets/scaled_train.csv')#[:5] #debug
		cross_validation_image_ids = [
			image_id 
			for index, image_id 
			in enumerate(all_image_ids)
			if index%4 == 0]
		greyscale_image_paths = [
			os.path.join('../Datasets/preprocessed_train_images', image_id + '.png')
			for image_id in cross_validation_image_ids]
		color_image_paths = [
			os.path.join('../Datasets/Color/preprocessed_train_images', image_id + '.png')
			for image_id in cross_validation_image_ids]
		unscaled_images_dir = '../Datasets/train_images'

		GenerateSubmissionFile(
			cross_validation_image_ids, 
			greyscale_image_paths,
			color_image_paths, 
			unscaled_images_dir,
			'CrossValidationSetPredictions.csv', 
			include_confidence = False)

	if RUN_ON_FINAL_TEST_IMAGES:
		greyscale_final_test_data_dir_path = '../Datasets/preprocessed_test_images'
		image_file_names = os.listdir(greyscale_final_test_data_dir_path)#[:5] #debug
		greyscale_image_paths = [
			os.path.join(greyscale_final_test_data_dir_path, file_name) 
			for file_name in image_file_names]

		color_final_test_data_dir_path = '../Datasets/Color/preprocessed_test_images'
		color_image_paths = [
			os.path.join(color_final_test_data_dir_path, file_name)
			for file_name in image_file_names]
		
		image_ids = [file_name.replace('.png', '') for file_name in image_file_names]
		unscaled_images_dir = '../Datasets/test_images'

		GenerateSubmissionFile(
			image_ids, 
			greyscale_image_paths, 
			color_image_paths,
			unscaled_images_dir,
			'FinalTestSetPredictions.csv', 
			include_confidence = False)