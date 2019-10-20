import torch.nn as nn

class RPN(nn.Module):
	def __init__(self, input_channels, anchor_count):
		super(RPN, self).__init__()

		self.ContainsObjectClassifier = nn.Conv2d(
			in_channels = input_channels, 
			out_channels = 2*anchor_count, 
			kernel_size = 1, 
			stride = 1, 
			padding = 0)
		self.RegionRegressor = nn.Conv2d(
			in_channels = input_channels, 
			out_channels = 4*anchor_count, 
			kernel_size = 1, 
			stride = 1, 
			padding = 0)

	def forward(self, features):
		class_predictions = self.ContainsObjectClassifier(features)
		original_class_predictions_shape = class_predictions.shape
		class_predictions = class_predictions.view((
			original_class_predictions_shape[0], 	# Batch size
			2,										# Class predictions (per anchor)
			original_class_predictions_shape[1]//2,	# Anchor count
			original_class_predictions_shape[2],	# Feature map width
			original_class_predictions_shape[3]))	# Feature map height
		
		region_predictions = self.RegionRegressor(features)
		original_region_predictions_shape = region_predictions.shape
		region_predictions = region_predictions.view((
			original_region_predictions_shape[0], 	# Batch size
			4,										# Bounding box regression outputs (per anchor)
			original_region_predictions_shape[1]//4,# Anchor count
			original_region_predictions_shape[2],	# Feature map width
			original_region_predictions_shape[3]))	# Feature map height

		return class_predictions, region_predictions