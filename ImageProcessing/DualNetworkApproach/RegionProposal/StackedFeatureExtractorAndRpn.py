import torch.nn as nn

class StackedFeatureExtractorAndRpn(nn.Module):
	def __init__(self, feature_extractor, rpn_network):
		super(StackedFeatureExtractorAndRpn, self).__init__()
		self.FeatureExtractor = feature_extractor
		self.RpnNetwork = rpn_network

	def forward(self, image):
		features = self.FeatureExtractor(image)
		anchor_classes, bbox_regression_outputs = self.RpnNetwork(features)

		return anchor_classes, bbox_regression_outputs