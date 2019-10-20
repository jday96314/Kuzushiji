import torch.nn as nn

class DenseCharacterClassifier(nn.Module):
	def __init__(
			self, 
			input_feature_count, 
			hidden_sizes, 
			dropout_rates_after_hidden_layers, 
			num_classes,
			hidden_activation = 'ReLU'):
		super(DenseCharacterClassifier, self).__init__()

		hidden_layers = []
		input_dims = input_feature_count
		for hidden_layer_index, hidden_size in enumerate(hidden_sizes):
			hidden_layers.append(nn.Linear(input_feature_count, hidden_size))
			dropout_rate = dropout_rates_after_hidden_layers[hidden_layer_index]
			if dropout_rate > 0:
				hidden_layers.append(nn.Dropout(p = dropout_rate))

			if hidden_activation == 'ReLU':
				hidden_layers.append(nn.ReLU())
			elif hidden_activation == 'ELU':
				hidden_layers.append(nn.ReLU())
			elif hidden_activation == 'LeakyReLU':
				hidden_layers.append(nn.LeakyReLU(negative_slope=0.01))
			else:
				print('ERROR: Unsupported activation function in the dense character classifier!')
				exit()

		self.HiddenLayers = nn.Sequential(*hidden_layers)
		self.OutputLayer = nn.Linear(hidden_size, num_classes)

	def forward(self, features):
		x = self.HiddenLayers(features)
		x = self.OutputLayer(x)
		return x