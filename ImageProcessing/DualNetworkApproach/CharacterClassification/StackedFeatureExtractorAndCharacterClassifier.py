import torch.nn as nn
from torchvision.ops import RoIPool, RoIAlign

class StackedFeatureExtractorAndCharacterClassifier(nn.Module):
	def __init__(self, feature_extractor, pool_output_size, character_classifier, spatial_scale, dropout_rate = 0):
		super(StackedFeatureExtractorAndCharacterClassifier, self).__init__()
		self.FeatureExtractor = nn.parallel.DataParallel(feature_extractor)
		self.Dropout = nn.Dropout(p = dropout_rate)
		self.PoolOutputSize = pool_output_size
		self.RegionPool = RoIAlign(pool_output_size, spatial_scale, 1)

		self.CharacterClassifier = character_classifier

	# The images are expected to have the following shape:
	# 	(batch size, image channels, image width, image height)
	# This is noteworthy because by convention pytorch does 
	# 	(batch size, image channels, image height, image width)
	#
	# The ROIs need to have the following format. Note that the xs and ys are in an unintuitive format.
	# This is because pytorch expects the image dimensions to be a bit wonkey.
	# [
	#	[image_id_0, top_left_y_0, top_left_x_0, bottom_right_y_0, bottom_right_x_0],
	#	[image_id_0, top_left_y_1, top_left_x_1, bottom_right_y_1, bottom_right_x_1],
	#	...
	#	[image_id_1, ...],
	#	[image_id_1, ...],
	# 	...
	#	[image_id_n, top_left_y_n, top_left_x_n, bottom_right_y_n, bottom_right_x_n]	
	# ]
	#
	# The x/y orders in the ROIs could be made more intuitive by transposing the final two dimensions
	# of the input images. This actually wouldn't require any code changes here.
	def forward(self, image, rois):
		features = self.FeatureExtractor(image)
		features = self.Dropout(features)
		region_features = self.RegionPool(features, rois.view(-1, 5))
		region_features = region_features.view(-1, self.FeatureExtractor.module.FinalChannelsCount * self.PoolOutputSize[0] * self.PoolOutputSize[1]) # TODO: Double check that this is safe. The first dim of the input *must* be the ROI dim.
		character_classes = self.CharacterClassifier(region_features)

		return character_classes