import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PositionalEncodings import *

class BasicBlock(nn.Module):
	def __init__(self, embedding_dim_count, hidden_unit_count, hidden_layer_count, bidirectional, dropout_rate = 0):
		super(BasicBlock, self).__init__()
		self.InputDropout = nn.Dropout(p = dropout_rate//2)
		self.HiddenLayers = nn.GRU(
			input_size = embedding_dim_count, 
			hidden_size = hidden_unit_count, 
			num_layers = hidden_layer_count,
			bidirectional=bidirectional)
		self.HiddenDropout = nn.Dropout(p = dropout_rate//2)		
		self.Adjustor = nn.Linear(hidden_unit_count*(1+bidirectional), embedding_dim_count)

	def forward(self, embedded_labels):
		features, _ = self.HiddenLayers(self.InputDropout(embedded_labels))
		adjustments = self.Adjustor(self.HiddenDropout(features))
		out = embedded_labels + adjustments

		return out

class DualInputSingleModificationBlock(nn.Module):
	def __init__(self, sequence_dim_count, downsampled_sequence_dim_count, recurrrent_unit_count, dropout_rate):
		super(DualInputSingleModificationBlock, self).__init__()
		self.ReferenceSequenceDropout = nn.Dropout(p = dropout_rate//2)
		self.ReferenceSequenceDownsampler = nn.Linear(sequence_dim_count, downsampled_sequence_dim_count)

		self.SequenceToModifyDropout = nn.Dropout(p = dropout_rate//2)
		self.SequenceToModifyDownsampler = nn.Linear(sequence_dim_count, downsampled_sequence_dim_count)

		self.HiddenLayer = nn.GRU(input_size = 2*downsampled_sequence_dim_count, hidden_size = recurrrent_unit_count, bidirectional=True)
		self.HiddenDropout = nn.Dropout(p = dropout_rate//2)
		self.Adjustor = nn.Linear(2*recurrrent_unit_count, sequence_dim_count)

	def forward(self, reference_sequence, sequence_to_modify):
		downsampled_reference_sequence = self.ReferenceSequenceDownsampler(self.ReferenceSequenceDropout(reference_sequence))
		downsampled_sequence_to_modify = self.SequenceToModifyDownsampler(self.SequenceToModifyDropout(sequence_to_modify))
		merged_sequece = F.relu(torch.cat((downsampled_reference_sequence, downsampled_sequence_to_modify), dim = -1))

		features, _ = self.HiddenLayer(merged_sequece)
		adjustments = self.Adjustor(self.HiddenDropout(features))
		modified_sequence = sequence_to_modify + adjustments

		return modified_sequence

class DualInputDualModificationBlock(nn.Module):
	def __init__(self, sequence_dim_count, recurrrent_unit_count, recurrent_layer_count, dropout_rate):
		super(DualInputDualModificationBlock, self).__init__()
		self.InputDropout = nn.Dropout(p = dropout_rate//2)
		self.SpatialPositionalEncodingLayer = nn.Linear(2, sequence_dim_count//2)
		self.MainSequenceDownsampler = nn.Linear(sequence_dim_count, sequence_dim_count//2)
		self.SkippedCharsSequenceDownsampler = nn.Linear(sequence_dim_count, sequence_dim_count//2)

		self.MergedSequenceDropout = nn.Dropout(p = .5)

		self.HiddenLayer = nn.GRU(
			input_size = sequence_dim_count, 
			num_layers=recurrent_layer_count,
			dropout=dropout_rate//8,
			hidden_size = recurrrent_unit_count, 
			bidirectional=True)

		self.HiddenDropout = nn.Dropout(p = dropout_rate//2)
		# self.MainSequenceAdjustor = nn.Linear(2*recurrrent_unit_count, sequence_dim_count)
		# self.SkippedCharsSequenceAdjustor = nn.Linear(2*recurrrent_unit_count, sequence_dim_count)

		# dense_hidden_unit_count = int(.5*(2*recurrrent_unit_count + sequence_dim_count))
		# self.MainSequenceAdjustorHidden = nn.Linear(2*recurrrent_unit_count, dense_hidden_unit_count)
		self.MainSequenceAdjustor = nn.Linear(2*recurrrent_unit_count, sequence_dim_count)
		# self.SkippedCharsSequenceAdjustorHidden = nn.Linear(2*recurrrent_unit_count, dense_hidden_unit_count)
		self.SkippedCharsSequenceAdjustor = nn.Linear(2*recurrrent_unit_count, sequence_dim_count)


		self.GruInitialState = nn.Parameter(
			torch.randn(recurrent_layer_count*2, 1, recurrrent_unit_count).type(torch.FloatTensor), 
			requires_grad=True)

	def forward(self, location_sequence, main_sequence, skipped_chars_sequence):
		downsampled_main_sequence = self.MainSequenceDownsampler(self.InputDropout(main_sequence))
		embedded_main_sequence_locations = self.SpatialPositionalEncodingLayer(location_sequence)
		downsampled_skipped_chars_sequence = self.SkippedCharsSequenceDownsampler(self.InputDropout(skipped_chars_sequence))
		merged_sequence = torch.cat((downsampled_main_sequence + embedded_main_sequence_locations, downsampled_skipped_chars_sequence), dim = -1)

		merged_sequence = self.MergedSequenceDropout(merged_sequence)

		# embedded_main_sequence_locations = self.SpatialPositionalEncodingLayer(location_sequence)
		# merged_sequence = torch.cat([embedded_main_sequence_locations, main_sequence, skipped_chars_sequence], dim = -1)
		# merged_sequence = self.UnifiedDownsampler(self.InputDropout(merged_sequence))

		grus_count = self.GruInitialState.shape[0]
		current_batch_size = main_sequence.shape[1]
		h0 = torch.stack([self.GruInitialState for _ in range(current_batch_size)], dim=1)
		h0 = h0.view(grus_count, current_batch_size, -1)
		features, _ = self.HiddenLayer(merged_sequence, h0)

		main_sequence_adjustments = self.MainSequenceAdjustor(self.HiddenDropout(features))
		modified_main_sequence = main_sequence + main_sequence_adjustments

		skipped_chars_sequence_adjustments = self.SkippedCharsSequenceAdjustor(self.HiddenDropout(features))
		modified_skipped_chars_sequence = skipped_chars_sequence + skipped_chars_sequence_adjustments		

		# main_sequence_adjustments = self.MainSequenceAdjustor(F.relu(self.MainSequenceAdjustorHidden(self.HiddenDropout(features))))
		# modified_main_sequence = main_sequence + main_sequence_adjustments

		# skipped_chars_sequence_adjustments = self.SkippedCharsSequenceAdjustor(F.relu(self.MainSequenceAdjustorHidden(self.HiddenDropout(features))))
		# modified_skipped_chars_sequence = skipped_chars_sequence + skipped_chars_sequence_adjustments		

		return modified_main_sequence, modified_skipped_chars_sequence

class ClassificationCorrector(nn.Module):
	def __init__(self, char_label_class_count, embedding_dim_count):
		super(ClassificationCorrector, self).__init__()
		self.FullEmbeddingDimCount = embedding_dim_count
		self.CharEmbeddingLayer = nn.Embedding(char_label_class_count, embedding_dim_count)
		self.ConfidenceEmbeddingLayer = nn.Linear(1, embedding_dim_count)
		self.SpatialPositionalEncodingLayer = nn.Linear(2, embedding_dim_count)

		self.SkippedCharsSequenceCreator = nn.GRU(embedding_dim_count, 512, 2, bidirectional=True, dropout=.25)
		self.SkippedCharsSequenceFormattor = nn.Linear(512*2, embedding_dim_count)
		self.AdjustmentBlocks = nn.ModuleList([DualInputDualModificationBlock(embedding_dim_count, 512, 2, 1).cuda() for i in range(12)])
		# self.ResidualConnectionDropout = nn.Dropout(.0625)

		char_type_hidden_dim_count = int(.5 * (embedding_dim_count + char_label_class_count))
		self.CharLabelDropout = nn.Dropout(.25)
		self.CharLabelHidden = nn.Linear(embedding_dim_count, char_type_hidden_dim_count)
		self.CharLabelOutput = nn.Linear(char_type_hidden_dim_count, char_label_class_count)
		self.CharLocationDropout = nn.Dropout(.25)
		self.CharLocationHidden = nn.Linear(embedding_dim_count, embedding_dim_count//2)
		self.CharLocationOutput = nn.Linear(embedding_dim_count//2, 2)

		# TODO: Experiment with adding hidden layers.
		self.SkippedCharLabelDropout = nn.Dropout(.25)
		self.SkippedCharLabelHidden = nn.Linear(embedding_dim_count, char_type_hidden_dim_count)
		self.SkippedCharLabelOputput = nn.Linear(char_type_hidden_dim_count, char_label_class_count)
		self.SkippedCharLocationDropout = nn.Dropout(.25)
		self.SkippedCharLocationHidden = nn.Linear(embedding_dim_count, embedding_dim_count//2)
		self.SkippedCharLocationOutput = nn.Linear(embedding_dim_count//2, 2)

	# input_character_locations should have shape (sequence len, batch size, 2)
	# input_character_labels should have shape (sequence len, batch size, 1)
	def forward(
			self, 
			input_character_locations, 
			input_character_labels,
			input_character_confidences):
		# CREATE THE EMBEDDINGS.
		embedded_input_labels = self.CharEmbeddingLayer(input_character_labels).view(
			input_character_labels.shape[0], input_character_labels.shape[1], -1)
		# embedded_input_labels *= (input_character_confidences)
		embedded_input_labels += self.ConfidenceEmbeddingLayer(input_character_confidences)

		# TRANSFORM THE SEQUENCE.
		transformed_sequence = embedded_input_labels
		embedded_char_locations = self.SpatialPositionalEncodingLayer(input_character_locations)
		skipped_chars_sequence, _ = self.SkippedCharsSequenceCreator(embedded_input_labels + embedded_char_locations)
		skipped_chars_sequence = self.SkippedCharsSequenceFormattor(skipped_chars_sequence)
		for layer_index, layer in enumerate(self.AdjustmentBlocks):
			transformed_sequence, skipped_chars_sequence = layer(input_character_locations, transformed_sequence, skipped_chars_sequence)
			# transformed_sequence = self.ResidualConnectionDropout(transformed_sequence)
			# skipped_chars_sequence = self.ResidualConnectionDropout(skipped_chars_sequence)

		# DECODE THE TRANSFORMED SEQUENCE.
		transformed_char_types_hidden = F.relu(self.CharLabelHidden(self.CharLabelDropout(transformed_sequence)))
		transformed_char_types = F.log_softmax(self.CharLabelOutput(transformed_char_types_hidden), dim = -1)
		char_locations_hidden = torch.relu(self.CharLocationHidden(self.CharLocationDropout(transformed_sequence)))
		char_locations = torch.tanh(self.CharLocationOutput(char_locations_hidden))

		skipped_char_types_hidden = F.relu(self.SkippedCharLabelHidden(self.SkippedCharLabelDropout(skipped_chars_sequence)))
		skipped_char_labels = F.log_softmax(self.SkippedCharLabelOputput(skipped_char_types_hidden), dim = -1)
		skipped_char_locations_hidden = torch.relu(self.SkippedCharLocationHidden(self.SkippedCharLocationDropout(skipped_chars_sequence)))
		skipped_char_locations = torch.tanh(self.SkippedCharLocationOutput(skipped_char_locations_hidden))

		return transformed_char_types, char_locations, skipped_char_labels, skipped_char_locations