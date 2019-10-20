import copy
import random

class ObjectDetectionsMangler:
	# -	normalized_confusion_matrix[ground_truth_ord][predicted_ord] = probability of characters with
	#	type ground_truth_ord being labeled as type predicted_ord
	# - label_location_x_stddev is the standard deviation of the distance between a character's real
	#	location and its detected location (in the x direction). This is expected to be a fraction of 
	#	the original image width (e.g. if the standard dev is 10% of the image width then this should 
	#	be .1).
	# - label_location_y_stddev is like label_location_x_stddev but in the y direction and represented
	#	as a fraction of the image height.
	# - mean_extraneous_characters_per_image is the mean number of extraneous charaters that are detected
	#	in each image.
	# - stddev_extraneous_characters_per_image is the standard deviation of the number of extraneous
	#	characters that are detected in each image.
	# - extraneous_character_type_probs[ord] is the probability that an extraneous character is predicted
	#	to have that ordinal value.
	# - character_omission_probability is the probability of a character failing to be detected.
	def __init__(
			self, 
			normalized_confusion_matrix, 
			label_location_x_stddev,
			label_location_y_stddev,
			mean_extraneous_characters_per_image,
			stddev_extraneous_characters_per_image,
			extraneous_character_type_probs,
			character_omission_probability):
		self.NormalizedConfusionMatrix = normalized_confusion_matrix
		self.LabelLocationXStddev = label_location_x_stddev
		self.LabelLocationYStddev = label_location_y_stddev
		self.MeanExtraneousCharactersPerImage = mean_extraneous_characters_per_image
		self.StddevExtraneousCharactersPerImage = stddev_extraneous_characters_per_image
		self.ExtraneousCharacterTypeProbs = extraneous_character_type_probs
		self.CharacterOmissionProbability = character_omission_probability

	def GetMangledSequence(self, ground_truth_locations, ground_truth_char_labels):
		mangled_locations = copy.deepcopy(ground_truth_locations)
		mangled_char_labels = copy.deepcopy(ground_truth_char_labels)
		self.MangleCharacterTypes(mangled_char_labels)
		self.AddLabelLocationJitter(mangled_locations)
		self.OmitLabels(mangled_locations, mangled_char_labels)
		self.AddExtraneousLabels(mangled_locations, mangled_char_labels)
		self.ReorderCharacters(mangled_locations, mangled_char_labels)
		return mangled_locations, mangled_char_labels

	def MangleCharacterTypes(self, char_sequence_to_mangle):
		for char in char_sequence_to_mangle:
			if char[0] <= 2: # EOS, SOS, and Pad should not be modified.
				continue
			if random.random() < .1:
				char[0] = int(random.random() * 2000) + 3
		pass

	def AddLabelLocationJitter(self, location_sequence_to_mangle):
		pass

	def OmitLabels(self, char_sequence_to_mangle, location_sequence_to_mangle):
		pass

	def AddExtraneousLabels(self, char_sequence_to_mangle, location_sequence_to_mangle):
		pass

	def ReorderCharacters(self, char_sequence_to_order, location_sequence_to_order):
		# Should call into GetLabelsSortedByReadOrder
		pass
