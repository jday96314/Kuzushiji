import torch.nn as nn

class VGG(nn.Module):
	def __init__(self, image_channels, layer_configs):
		super(VGG, self).__init__()

		# CREATE THE NETWORK'S LAYERS.
		layers = []
		in_channels = image_channels
		for layer_config in layer_configs:
			if layer_config == 'MaxPool':
				layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
			elif layer_config == 'BatchNorm':
				layers.append(nn.BatchNorm2d(out_channels))
			else:
				out_channels = layer_config
				layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
				layers.append(nn.ReLU(inplace=True))
				#layers.append(nn.LeakyReLU(.01, inplace=True))
				self.FinalChannelsCount = out_channels
				in_channels = out_channels

		self.FeatureCreator = nn.Sequential(*layers)

		# INITIALIZE THE NETWORK'S PARAMETERS.
		for layer in self.FeatureCreator:
			if isinstance(layer, nn.Conv2d):
				nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
				if layer.bias is not None:
					nn.init.constant_(layer.bias, 0)

	def forward(self, x):
		features = self.FeatureCreator(x)
		return features