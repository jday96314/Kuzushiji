import copy
import random
import numpy as np
import torch
import math
from CharSequencesDataset import GetLabelsSortedByReadOrder

class ObjectDetectionsMangler:
	def __init__(
			self, 
			error_stats,
			reserved_char_count,
			not_a_real_char_label_id,
			boost_drop_probs = False):
		# # a matrix C in which C[i][j] is the number of characters with char index i that were labeled as having
		# # char index j. The indices align with where chars are located in the sorted charset.
		# # These indices do not account for the reserved chars.
		# self.ConfusionMatrix = error_stats['ConfusionMatrix']

		char_classes_count = len(error_stats['ConfusionMatrix'])
		self.CharIdsWithOffset = [i + reserved_char_count for i in range(char_classes_count)]
		default_probs = np.array([1/char_classes_count for i in range(char_classes_count)])
		self.NormalizedConfusionMatrix = [
			error_stats['ConfusionMatrix'][i] / sum(error_stats['ConfusionMatrix'][i]) if sum(error_stats['ConfusionMatrix'][i]) > 0 else default_probs
			for i in range(len(error_stats['ConfusionMatrix']))]

		# The mean horizontal distance between a char's predicted location and its actual location.
		# E[ground_truth - predicted]
		self.MeanXOffset = error_stats['MeanXOffset']
		# The mean vertical distance between a char's predicted location and its actual location.
		# E[ground_truth - predicted]
		self.MeanYOffset = error_stats['MeanYOffset']
		# The standard deviation of the vertical distances between a char's predicted location and its actual 
		# location.
		# sqrt(var(ground_truth - predicted))
		self.StdYOffset = error_stats['StdYOffset']
		# The standard deviation of the horizontal distances between a char's predicted location and its actual 
		# location.
		# sqrt(var(ground_truth - predicted))
		self.StdXOffset = error_stats['StdXOffset']
		# The probability of an image containing at least one extraneous character.
		self.ProbImageContainsExtraneousChar = error_stats['ProbImageContainsExtraneousChar']
		# The mean number of extraneous characters in each image.
		self.MeanImageExtraneousCharCount = error_stats['MeanImageExtraneousCharCount']
		# The standard deviation of the number of extraneous characters in each image.
		self.StdImageExtraneousCharCount = error_stats['StdImageExtraneousCharCount']
		# The probability of an extraneous char belonging to each char class.
		# These indices do not account for the reserved chars.
		self.ExtraneousCharClassProbs = error_stats['ExtraneousCharClassProbs']
		# The probability of each type of character failing to be detected.
		# These indices do not account for the reserved chars.
		self.FailedDetectionProbByCharClass = error_stats['FailedDetectionProbByCharClass']
		# The default probability of a char failing to be detected (ignoring its type).
		self.DefaultFailedDetectionProb = np.mean(self.FailedDetectionProbByCharClass)
		# The number of chars types that are reserved for SOS, EOS, pad, NA, etc.
		self.ReservedCharCount = reserved_char_count
		# The "ground truth" type for detected chars that aren't real.
		self.NotARealCharLabel = not_a_real_char_label_id

		self.MeanCorrectLabelConfidence = error_stats['MeanCorrectLabelConfidence']
		self.StdCorrectLabelConfidence = error_stats['StdCorrectLabelConfidence']
		self.MeanIncorrectLabelConfidence = error_stats['MeanIncorrectLabelConfidence']
		self.stdIncorrectLabelConfidence = error_stats['stdIncorrectLabelConfidence']
		self.MeanExtraneousLabelConfidence = error_stats['MeanExtraneousLabelConfidence']
		self.StdExtraneousLabelConfidence = error_stats['StdExtraneousLabelConfidence']
		
		# Indicates if characters should be skipped unrealisticly frequently.
		self.BoostDropProbs = boost_drop_probs

	def GetMangledSequence(self, ground_truth_locations, ground_truth_char_labels, image_width, image_height):
		ground_truth_locations_with_mangled_len = copy.deepcopy(ground_truth_locations)
		ground_truth_labels_with_mangled_len = copy.deepcopy(ground_truth_char_labels)
		detection_confidences = torch.ones((len(ground_truth_locations), 1), dtype=torch.float32)
		self.AddExtraneousLabels(ground_truth_locations_with_mangled_len, ground_truth_labels_with_mangled_len, detection_confidences)
		self.AddLabelLocationJitter(ground_truth_locations_with_mangled_len, image_width, image_height)
		self.ReorderCharacters(ground_truth_locations_with_mangled_len, ground_truth_labels_with_mangled_len, detection_confidences)

		(
			ground_truth_locations_with_mangled_len, 
			ground_truth_labels_with_mangled_len, 
			detection_confidences,
			skipped_char_locations, 
			skipped_char_labels,
		) = self.OmitLabels(
			ground_truth_locations_with_mangled_len, 
			ground_truth_labels_with_mangled_len,
			detection_confidences)
		mangled_char_labels = copy.deepcopy(ground_truth_labels_with_mangled_len)
		self.MangleCharacterTypes(mangled_char_labels, detection_confidences)

		return (
			ground_truth_locations_with_mangled_len, 
			ground_truth_labels_with_mangled_len,
			skipped_char_locations, 
			skipped_char_labels,
			ground_truth_locations_with_mangled_len,
			mangled_char_labels,
			detection_confidences)

	def OmitLabels(self, location_sequence_to_mangle, label_sequence_to_mangle, detection_confidences):
		if len(label_sequence_to_mangle) == 0:
			return location_sequence_to_mangle, label_sequence_to_mangle
		
		partial_location_sequence = []
		partial_label_sequence = []
		partial_confidence_sequence = []
		skipped_location_sequence = []
		skipped_label_sequence = []
		skipped_char_type = self.NotARealCharLabel
		skipped_char_location = None
		NO_OFFSET = torch.tensor([0, 0], dtype = torch.float32)

		for char_index in range(len(label_sequence_to_mangle)):
			if skipped_char_location is not None:
				raw_distance = skipped_char_location - location_sequence_to_mangle[char_index]
				raw_distance[0] = min(raw_distance[0], .05)
				raw_distance[1] = min(raw_distance[1], .05)
				skipped_location_sequence[-1] = raw_distance * 10 #skipped_char_location - location_sequence_to_mangle[char_index]
				skipped_label_sequence[-1] = torch.tensor([skipped_char_type], dtype = torch.long)
			else:
				skipped_location_sequence.append(NO_OFFSET)
				skipped_label_sequence.append(torch.tensor([skipped_char_type], dtype = torch.long))

			# RESERVED CHARS SHOULD NEVER BE DROPPED.
			char_type_id = label_sequence_to_mangle[char_index].numpy()[0]
			if char_type_id < self.ReservedCharCount:
				partial_location_sequence.append(location_sequence_to_mangle[char_index])
				partial_label_sequence.append(label_sequence_to_mangle[char_index])
				partial_confidence_sequence.append(detection_confidences[char_index])
				skipped_char_type = torch.tensor([self.NotARealCharLabel], dtype = torch.long)
				skipped_char_location = None
				continue

			p_drop_char = self.FailedDetectionProbByCharClass[char_type_id - self.ReservedCharCount]
			if p_drop_char == 0:
				p_drop_char = self.DefaultFailedDetectionProb * .5
			if self.BoostDropProbs and (random.random() > .5):
				p_drop_char *= 1 + 2*random.random()

			if random.random() < p_drop_char:
				skipped_char_type = char_type_id
				skipped_char_location = location_sequence_to_mangle[char_index]
				continue
			else:
				skipped_char_type = torch.tensor([self.NotARealCharLabel], dtype = torch.long)
				skipped_char_location = None

			partial_location_sequence.append(location_sequence_to_mangle[char_index])
			partial_label_sequence.append(label_sequence_to_mangle[char_index])
			partial_confidence_sequence.append(detection_confidences[char_index])

		return (
			torch.stack(partial_location_sequence), 
			torch.stack(partial_label_sequence), 
			torch.stack(partial_confidence_sequence),
			torch.stack(skipped_location_sequence),
			torch.stack(skipped_label_sequence))

	def MangleCharacterTypes(self, char_sequence_to_mangle, detection_confidences):
		for char_index, char in enumerate(char_sequence_to_mangle):
			# Reserved chars should not be modified.
			initial_char_type_id = copy.deepcopy(char[0])
			if initial_char_type_id == self.NotARealCharLabel:
				char[0] = int(np.random.choice(
					self.CharIdsWithOffset, 
					p =self.ExtraneousCharClassProbs))

			if not (initial_char_type_id < self.ReservedCharCount):
				char[0] = int(np.random.choice(
					self.CharIdsWithOffset, 
					p = self.NormalizedConfusionMatrix[initial_char_type_id - self.ReservedCharCount]))

				#print(initial_char_type_id, char[0], self.NormalizedConfusionMatrix[initial_char_type_id - self.ReservedCharCount].max())
				if initial_char_type_id == char[0]:
					confidence = np.random.normal(self.MeanCorrectLabelConfidence, self.StdCorrectLabelConfidence*1.2)
					confidence = min(confidence, 1)
					confidence = max(confidence, 0)
					detection_confidences[char_index] = confidence
				else:
					confidence = np.random.normal(self.MeanIncorrectLabelConfidence, self.stdIncorrectLabelConfidence*1.2)
					confidence = min(confidence, 1)
					confidence = max(confidence, 0)
					detection_confidences[char_index] = confidence

	def AddLabelLocationJitter(self, location_sequence_to_mangle, image_width, image_height):
		horizontal_offset_distribution = torch.distributions.normal.Normal(
			self.MeanXOffset / image_width,
			self.StdXOffset / image_width)
		horizontal_offsets = horizontal_offset_distribution.sample((
			location_sequence_to_mangle.shape[0],
			1))
		vertical_offset_distribution = torch.distributions.normal.Normal(
			self.MeanYOffset / image_height,
			self.StdYOffset / image_height)
		vertical_offsets = vertical_offset_distribution.sample((
			location_sequence_to_mangle.shape[0],
			1))
		offsets = torch.cat([horizontal_offsets, vertical_offsets], dim=-1)
		location_sequence_to_mangle += offsets

	def AddExtraneousLabels(self, location_sequence_to_mangle, label_sequence_to_mangle, detection_confidences):
		extraneous_chars_to_add_count = max(
			0, 
			round(np.random.normal(
				self.MeanImageExtraneousCharCount, 
				self.StdImageExtraneousCharCount)))
		if extraneous_chars_to_add_count == 0:
			return

		extraneous_char_location_distribution = torch.distributions.uniform.Uniform(0, 1)
		extraneous_char_locations = extraneous_char_location_distribution.sample((extraneous_chars_to_add_count, 2))
		location_sequence_to_mangle = torch.cat(
			[location_sequence_to_mangle, extraneous_char_locations])

		extraneous_char_labels = torch.ones(
			(extraneous_chars_to_add_count, 1), 
			dtype=torch.long) * self.NotARealCharLabel
		label_sequence_to_mangle = torch.cat(
			[label_sequence_to_mangle, extraneous_char_labels])

		extraneous_chars_confidences_distribution = torch.distributions.normal.Normal(
			self.MeanExtraneousLabelConfidence, 
			self.StdExtraneousLabelConfidence)
		extraneous_chars_confidences = extraneous_chars_confidences_distribution.sample((extraneous_chars_to_add_count, 1))
		extraneous_chars_confidences = torch.clamp(extraneous_chars_confidences, 0, 1)
		detection_confidences = torch.cat([detection_confidences, extraneous_chars_confidences])

	def ReorderCharacters(self, location_sequence_to_order, label_sequence_to_order, detection_confidences):
		location_sequence_to_order_np, label_sequence_to_order = GetLabelsSortedByReadOrder(
			location_sequence_to_order.numpy(), 
			label_sequence_to_order.numpy())
		# This is an ineficient hack. Ideally this should be done in the previous call.
		location_sequence_to_order_np, detection_confidences = GetLabelsSortedByReadOrder(
			location_sequence_to_order.numpy(), 
			detection_confidences.numpy())

		location_sequence_to_order = torch.tensor(location_sequence_to_order_np, dtype = torch.float32)
		label_sequence_to_order = torch.tensor(label_sequence_to_order, dtype = torch.long)
		detection_confidences = torch.tensor(detection_confidences, dtype = torch.float32)

if __name__ == '__main__':
	fail_probs = [0 for i in range(100)]
	fail_probs[1] = 1
	fail_probs[2] = 1
	mangler = ObjectDetectionsMangler(
		error_stats = {
			'ConfusionMatrix': torch.tensor([[1,1],[1,1]]),
			'MeanXOffset': 0,
			'MeanYOffset': 0,
			'StdXOffset':0,
			'StdYOffset':0,
			'ProbImageContainsExtraneousChar': 0,
			'MeanImageExtraneousCharCount': 0,
			'StdImageExtraneousCharCount': 0,
			'ExtraneousCharClassProbs': 0,
			'FailedDetectionProbByCharClass': fail_probs},
		reserved_char_count = 4,
		not_a_real_char_label_id = 3
	)
	mangler.DefaultFailedDetectionProb = 0
	
	chars = torch.tensor([[i + 4] for i in range(10)])
	locs = torch.tensor([[1.1*i,1.1*i] for i in range(10)])
	print(locs.shape)
	print(chars.shape)
	print()

	locs, chars, skipped_locs, skipped_chars = mangler.OmitLabels(locs, chars)
	print(locs.shape)
	print(chars.shape)
	print()
	print(skipped_locs.shape)
	print(skipped_chars.shape)

	for i in range(len(chars)):
		print(skipped_chars[i].numpy(), chars[i].numpy())