import torch.nn as nn

class RPN_WithHidden(nn.Module):
	def __init__(
			self, 
			input_channels, 
			anchor_count, 
			classifier_dropout_rate,
			regression_dropout_rate,
			classifier_hidden_units = 256, 
			regressor_hidden_units = 256):
		super(RPN_WithHidden, self).__init__()

		self.ClassifierDropout = nn.Dropout(p = classifier_dropout_rate)
		self.ClassifierHidden = nn.Conv2d(
			in_channels = input_channels, 
			out_channels = classifier_hidden_units, 
			kernel_size = 1, 
			stride = 1, 
			padding = 0)
		self.ContainsObjectClassifier = nn.Conv2d(
			in_channels = classifier_hidden_units, 
			out_channels = 2*anchor_count, 
			kernel_size = 1,
			stride = 1, 
			padding = 0)

		self.RegressorDropout = nn.Dropout(p = regression_dropout_rate)
		self.RegressorHidden = nn.Conv2d(
			in_channels = input_channels, 
			out_channels = regressor_hidden_units, 
			kernel_size = 1, 
			stride = 1, 
			padding = 0)
		self.RegionRegressor = nn.Conv2d(
			in_channels = regressor_hidden_units, 
			out_channels = 4*anchor_count, 
			kernel_size = 1, 
			stride = 1, 
			padding = 0)

	def forward(self, features):
		class_predictions = self.ContainsObjectClassifier(
			nn.functional.relu(self.ClassifierHidden(self.ClassifierDropout(features))))
		original_class_predictions_shape = class_predictions.shape
		class_predictions = class_predictions.view((
			original_class_predictions_shape[0], 	# Batch size
			2,										# Class predictions (per anchor)
			original_class_predictions_shape[1]//2,	# Anchor count
			original_class_predictions_shape[2],	# Feature map width
			original_class_predictions_shape[3]))	# Feature map height
		
		region_predictions = self.RegionRegressor(
			nn.functional.relu(self.RegressorHidden(self.RegressorDropout(features))))
		original_region_predictions_shape = region_predictions.shape
		region_predictions = region_predictions.view((
			original_region_predictions_shape[0], 	# Batch size
			4,										# Bounding box regression outputs (per anchor)
			original_region_predictions_shape[1]//4,# Anchor count
			original_region_predictions_shape[2],	# Feature map width
			original_region_predictions_shape[3]))	# Feature map height

		return class_predictions, region_predictions