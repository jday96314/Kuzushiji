import torch
import torch.nn as nn
import torch.nn.functional as F

from PositionalEncodings import *

class NextCharacterPredictor(nn.Module):
	def __init__(self, char_label_class_count, embedding_dim_count):
		super(NextCharacterPredictor, self).__init__()
		self.FullEmbeddingDimCount = embedding_dim_count
		self.CharEmbeddingLayer = nn.Embedding(char_label_class_count, embedding_dim_count)
		self.SequencePositionalEncodingLayer = PositionalEncoder(embedding_dim_count)
		self.SpatialPositionalEncodingLayer = nn.Linear(2, embedding_dim_count)

		encoder_layer = nn.TransformerEncoderLayer(
			d_model = embedding_dim_count, 
			nhead = 8, 
			dim_feedforward = 2048, 
			dropout = .1)
		self.TransformerEncoder = nn.TransformerEncoder(
			encoder_layer = encoder_layer,
			num_layers = 3)

		decoder_layer = nn.TransformerDecoderLayer(
			d_model = embedding_dim_count,
			nhead = 8,
			dim_feedforward = 2048,
			dropout = .1)
		self.TransformerDecoder = nn.TransformerDecoder(
			decoder_layer = decoder_layer,
			num_layers = 3)

		self.CharLabelOutput = nn.Linear(embedding_dim_count, char_label_class_count)
		self.NextCharLocationOutput = nn.Linear(embedding_dim_count, 2)
		self.NextCharLabelOutput = nn.Linear(embedding_dim_count, char_label_class_count)

	def init_weights(self):
		initrange = .1
		self.CharEmbeddingLayer.weight.uniform_(-initrange, initrange)
		self.CharEmbeddingLayer.bias.data.zero_()
		self.SpatialPositionalEncodingLayer.data.uniform_(-initrange, initrange)
		self.CharLocationOutput.weight.data.uniform_(-initrange, initrange)
		self.CharLabelOutput.weight.data.uniform_(-initrange, initrange)

	def _generate_square_subsequent_mask(self, dim_count):
		mask = (torch.triu(torch.ones(dim_count, dim_count)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda()
		return mask

	# input_character_locations should have shape (sequence len, batch size, 2)
	# input_character_labels should have shape (sequence len, batch size, 1)
	def forward(
			self, 
			input_character_locations, 
			input_character_labels):
		# CREATE THE EMBEDDINGS.
		embedded_input_labels = self.CharEmbeddingLayer(input_character_labels).view(
			input_character_labels.shape[0], input_character_labels.shape[1], -1)
		embedded_input_labels += self.SpatialPositionalEncodingLayer(input_character_locations)
		# embedded_input_labels *= math.sqrt(self.FullEmbeddingDimCount)
		# embedded_input_labels = self.SequencePositionalEncodingLayer(embedded_input_labels)

		# TRANSFORM THE SEQUENCE.
		encoded_labels = self.TransformerEncoder(
			src = embedded_input_labels)
		transformed_labels = self.TransformerDecoder(
			tgt = embedded_input_labels,
			memory = encoded_labels)

		# DECODE THE TRANSFORMED SEQUENCE.
		transformed_char_types = F.log_softmax(self.CharLabelOutput(transformed_labels), dim = -1)
		
		next_char_locations = self.NextCharLocationOutput(transformed_labels)
		next_char_labels = F.log_softmax(self.NextCharLabelOutput(transformed_labels), dim = -1)

		return transformed_char_types, next_char_locations, next_char_labels
