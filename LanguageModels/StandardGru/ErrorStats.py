from multiprocessing import Pool
import pandas as pd
import numpy as np
import pickle
import math

# preds_indices = {
# 	'label': 0,
# 	'X': 1,
# 	'Y': 2
# }

preds_indices = {
	'label': 0,
	'confidence': 1,
	'X': 2,
	'Y': 3
}

# Returns a matrix C in which C[i][j] is the number of characters with char index i that were labeled as having
# char index j. The indices align with where chars are located in the sorted charset.
def GenerateConfusionMatrixForImage(preds, truth, sorted_charset):
	char_class_count = len(sorted_charset)
	confusion_matrix = np.zeros(shape = (char_class_count, char_class_count))
	if str(truth) == 'nan' or str(preds) == 'nan':
		return confusion_matrix

	# PARSE THE GROUND TRUTH LABELS. 
	truth_indices = {
		'label': 0,
		'X': 1,
		'Y': 2,
		'Width': 3,
		'Height': 4
	}
	truth = truth.split(' ')
	truth_labels = np.array(truth[truth_indices['label']::len(truth_indices)])
	truth_xmins = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
	truth_ymins = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
	truth_xmaxes = truth_xmins + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
	truth_ymaxes = truth_ymins + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

	# PARSE THE PREDICTED LABELS.
	preds = preds.split(' ')
	preds_labels = np.array(preds[preds_indices['label']::len(preds_indices)])
	preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
	preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)

	# GENERATE THE CONFUSION MATRIX.
	for char_index in range(len(truth_xmins)):
		ground_truth_xmin = truth_xmins[char_index]
		ground_truth_ymin = truth_ymins[char_index]
		ground_truth_xmax = truth_xmaxes[char_index]
		ground_truth_ymax = truth_ymaxes[char_index]
		preds_inside_bbox_mask = (
			(preds_x > ground_truth_xmin) & 
			(preds_y > ground_truth_ymin) &
			(preds_x < ground_truth_xmax) &
			(preds_y < ground_truth_ymax))
		
		predicted_labels = preds_labels[preds_inside_bbox_mask]
		ground_truth_label_type_index = np.argwhere(sorted_charset == truth_labels[char_index])[0][0]
		for predicted_label in predicted_labels:
			predicted_label_type_index = np.argwhere(sorted_charset == predicted_label)[0][0]
			confusion_matrix[ground_truth_label_type_index][predicted_label_type_index] += 1

	return confusion_matrix.astype(int)

def GenerateConfusionMatrixForImages(
		submission_file_contents, 
		ground_truth_file_contents, 
		sorted_charset, 
		image_ids,
		worker_count = 1):
	if worker_count > 1:
		images_per_batch = math.ceil(len(submission_image_ids)//worker_count)
		worker_args = [
			(submission_file_contents, ground_truth_file_contents, sorted_charset, submission_image_ids[i:i+images_per_batch])
			for i in range(0, len(submission_image_ids), images_per_batch)]
		with Pool(worker_count) as pool:
			# Recursvely operates in batches instead of mapping over the individual images because memory
			# consuption increases linearly with the length of worker_args.
			worker_confusion_matrices = np.array(pool.starmap(GenerateConfusionMatrixForImages, worker_args))
		all_images_confusion_matrix = np.sum(worker_confusion_matrices, axis = 0)
	else:
		char_class_count = len(sorted_charset)
		all_images_confusion_matrix = np.zeros(shape = (char_class_count, char_class_count))
		for image_id in image_ids:
			all_images_confusion_matrix += GenerateConfusionMatrixForImage(
				submission_file_contents[submission_file_contents['image_id'] == image_id]['labels'].values[0],
				ground_truth_file_contents[ground_truth_file_contents['image_id'] == image_id]['labels'].values[0],
				sorted_charset)
	
	return all_images_confusion_matrix

# Returns two parallel arrays. The first is the observed horizontal distances between the centers of the ground truth
# bounding boxes and the predictions with matching labels. The second is similar, but contains vertical distances.
# Any matching label within 20px of the bounding box edges will be taken into consideration.
def GetImageXYDistances(preds, truth):
	x_distances = []
	y_distances = []
	if str(truth) == 'nan' or str(preds) == 'nan':
		return x_distances, y_distances

	# PARSE THE GROUND TRUTH LABELS. 
	truth_indices = {
		'label': 0,
		'X': 1,
		'Y': 2,
		'Width': 3,
		'Height': 4
	}
	truth = truth.split(' ')
	truth_labels = np.array(truth[truth_indices['label']::len(truth_indices)])
	truth_xmins = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
	truth_ymins = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
	truth_xmaxes = truth_xmins + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
	truth_ymaxes = truth_ymins + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

	# PARSE THE PREDICTED LABELS.
	preds = preds.split(' ')
	preds_labels = np.array(preds[preds_indices['label']::len(preds_indices)])
	preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
	preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)

	# GENERATE THE CONFUSION MATRIX.
	for char_index in range(len(truth_xmins)):
		ground_truth_label = truth_labels[char_index]
		ground_truth_xmin = truth_xmins[char_index]
		ground_truth_ymin = truth_ymins[char_index]
		ground_truth_xmax = truth_xmaxes[char_index]
		ground_truth_ymax = truth_ymaxes[char_index]
		ALLOWED_MARGIN_PX = 20
		preds_inside_bbox_mask = (
			(preds_x > ground_truth_xmin - ALLOWED_MARGIN_PX) & 
			(preds_y > ground_truth_ymin - ALLOWED_MARGIN_PX) &
			(preds_x < ground_truth_xmax - ALLOWED_MARGIN_PX) &
			(preds_y < ground_truth_ymax - ALLOWED_MARGIN_PX) &
			(preds_labels == ground_truth_label))
		
		predicted_xs = preds_x[preds_inside_bbox_mask]
		predicted_ys = preds_y[preds_inside_bbox_mask]
		for predicted_label_index in range(len(predicted_xs)):
			x_distance = np.mean([ground_truth_xmin, ground_truth_xmax]) - predicted_xs[predicted_label_index]
			x_distances.append(x_distance)
			y_distance = np.mean([ground_truth_ymin, ground_truth_ymax]) - predicted_ys[predicted_label_index]
			y_distances.append(y_distance)

	return x_distances, y_distances


def GetAllImagesXYDistanceStats(submission_file_contents, ground_truth_file_contents):
	all_x_offsets = []
	all_y_offsets = []
	for image_id in submission_file_contents['image_id'].values:
		image_x_offsets, image_y_offsets = GetImageXYDistances(
			submission_file_contents[submission_file_contents['image_id'] == image_id]['labels'].values[0],
			ground_truth_file_contents[ground_truth_file_contents['image_id'] == image_id]['labels'].values[0])
		all_x_offsets += image_x_offsets
		all_y_offsets += image_y_offsets

	return (np.mean(all_x_offsets), np.std(all_x_offsets), np.mean(all_y_offsets), np.std(all_y_offsets))

def GetExtraneousCharCountsByTypeInImage(preds, truth, char_to_idx_lookup):
	occurrence_counts = np.zeros(len(char_to_idx_lookup.keys()))
	if str(truth) == 'nan' or str(preds) == 'nan':
		return occurrence_counts

	# PARSE THE GROUND TRUTH LABELS. 
	truth_indices = {
		'label': 0,
		'X': 1,
		'Y': 2,
		'Width': 3,
		'Height': 4
	}
	truth = truth.split(' ')
	truth_xmins = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
	truth_ymins = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
	truth_xmaxes = truth_xmins + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
	truth_ymaxes = truth_ymins + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

	# PARSE THE PREDICTED LABELS.
	preds = preds.split(' ')
	preds_labels = np.array(preds[preds_indices['label']::len(preds_indices)])
	preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
	preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)

	# CHECK IF EACH PRED WAS EXTRANEOUS.
	for pred_index in range(len(preds_x)):
		ACCEPTABLE_MARGIN_PX = 5
		relevant_char_mask = (
			(preds_x[pred_index] > truth_xmins - ACCEPTABLE_MARGIN_PX) &
			(preds_x[pred_index] < truth_xmaxes + ACCEPTABLE_MARGIN_PX) &
			(preds_y[pred_index] > truth_ymins - ACCEPTABLE_MARGIN_PX) &
			(preds_y[pred_index] < truth_ymaxes + ACCEPTABLE_MARGIN_PX))

		pred_is_extraneous = sum(relevant_char_mask) == 0
		if pred_is_extraneous:
			occurrence_counts[char_to_idx_lookup[preds_labels[pred_index]]] += 1

	return occurrence_counts

# Returns:
#	- Prob of image containing at least one extreaneous char
#	- Mean extraneous char count
# 	- Stddev extraneous char counr
# 	- Prob of an extraneous char having each type. 
def GetAllImagesExtraneousCharStats(submission_file_contents, ground_truth_file_contents, sorted_charset):
	char_to_idx_lookup = {c:i for i,c in enumerate(sorted_charset)}
	images_with_extraneous_chars_count = 0
	extraneous_char_occurrence_counts_by_type = np.zeros(len(sorted_charset))
	extraneous_char_counts = []
	for image_id in submission_file_contents['image_id'].values:
		image_extraneous_char_occurrence_counts = GetExtraneousCharCountsByTypeInImage(
			submission_file_contents[submission_file_contents['image_id'] == image_id]['labels'].values[0],
			ground_truth_file_contents[ground_truth_file_contents['image_id'] == image_id]['labels'].values[0],
			char_to_idx_lookup)

		extraneous_char_count = sum(image_extraneous_char_occurrence_counts)
		extraneous_char_counts.append(extraneous_char_count)
		if extraneous_char_count >= 1:
			images_with_extraneous_chars_count += 1
			extraneous_char_occurrence_counts_by_type += image_extraneous_char_occurrence_counts

	return (
		# p(image has at least one extraneous char)
		images_with_extraneous_chars_count / len(submission_file_contents['image_id'].values),
		np.mean(extraneous_char_counts),
		np.std(extraneous_char_counts),
		# Prob of an extraneous char having each label type.
		extraneous_char_occurrence_counts_by_type / sum(extraneous_char_occurrence_counts_by_type))

# Returns the number of times each type of character was not detected. A character is considered to have not been
# detected if nothing was detected within 5px of the outer edge of its bounding box.
def GetImageFailedDetectionCounts(preds, truth, sorted_charset):
	failed_detection_counts = np.zeros(len(sorted_charset))
	occurrence_counts = np.zeros(len(sorted_charset))
	if str(truth) == 'nan' or str(preds) == 'nan':
		return failed_detection_counts, occurrence_counts

	# PARSE THE GROUND TRUTH LABELS. 
	truth_indices = {
		'label': 0,
		'X': 1,
		'Y': 2,
		'Width': 3,
		'Height': 4
	}
	truth = truth.split(' ')
	truth_labels = np.array(truth[truth_indices['label']::len(truth_indices)])
	truth_xmins = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
	truth_ymins = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
	truth_xmaxes = truth_xmins + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
	truth_ymaxes = truth_ymins + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

	# PARSE THE PREDICTED LABELS.
	preds = preds.split(' ')
	preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
	preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)

	for ground_truth_char_index in range(len(truth_labels)):
		ACCEPTABLE_MARGIN_PX = 5
		ground_truth_xmin = truth_xmins[ground_truth_char_index]
		ground_truth_ymin = truth_ymins[ground_truth_char_index]
		ground_truth_xmax = truth_xmaxes[ground_truth_char_index]
		ground_truth_ymax = truth_ymaxes[ground_truth_char_index]
		preds_inside_bbox_mask = (
			(preds_x > ground_truth_xmin - ACCEPTABLE_MARGIN_PX) & 
			(preds_y > ground_truth_ymin - ACCEPTABLE_MARGIN_PX) &
			(preds_x < ground_truth_xmax + ACCEPTABLE_MARGIN_PX) &
			(preds_y < ground_truth_ymax + ACCEPTABLE_MARGIN_PX))

		label_type_index = np.argwhere(sorted_charset == truth_labels[ground_truth_char_index])[0][0]
		occurrence_counts[label_type_index] += 1
		char_was_detected = sum(preds_inside_bbox_mask) > 0
		if not char_was_detected:
			failed_detection_counts[label_type_index] += 1

	return failed_detection_counts, occurrence_counts

def GetAllImagesFailedDetectionProbs(submission_file_contents, ground_truth_file_contents, sorted_charset):
	all_images_failed_detection_counts = np.zeros(len(sorted_charset))
	all_images_char_occurrence_counts = np.zeros(len(sorted_charset))
	for image_id in submission_file_contents['image_id'].values:
		failed_detection_counts, char_occurrence_counts = GetImageFailedDetectionCounts(
			submission_file_contents[submission_file_contents['image_id'] == image_id]['labels'].values[0],
			ground_truth_file_contents[ground_truth_file_contents['image_id'] == image_id]['labels'].values[0],
			sorted_charset)

		all_images_failed_detection_counts += failed_detection_counts
		all_images_char_occurrence_counts += char_occurrence_counts

	placeholder_failed_detection_prob = sum(all_images_failed_detection_counts) / sum(all_images_char_occurrence_counts)
	failed_detection_probs = [
		all_images_failed_detection_counts[i]/all_images_char_occurrence_counts[i] if all_images_char_occurrence_counts[i] > 0 else placeholder_failed_detection_prob
		for i in range(len(all_images_char_occurrence_counts))]

	return failed_detection_probs

# Returns 
# (
#	all confidences from correct labels,
#	all confidences from incorrect labels,
#	all confidences from extraneous labels
# )
def GetConfidencesByTypeForImage(preds, truth):
	correct_pred_confidences = []
	incorrect_pred_confidences = []
	extraneous_pred_confidences = []
	if str(truth) == 'nan' or str(preds) == 'nan':
		return correct_pred_confidences, incorrect_pred_confidences, extraneous_pred_confidences

	# PARSE THE GROUND TRUTH LABELS. 
	truth_indices = {
		'label': 0,
		'X': 1,
		'Y': 2,
		'Width': 3,
		'Height': 4
	}
	truth = truth.split(' ')
	truth_labels = np.array(truth[truth_indices['label']::len(truth_indices)])
	truth_xmins = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
	truth_ymins = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
	truth_xmaxes = truth_xmins + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
	truth_ymaxes = truth_ymins + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

	# PARSE THE PREDICTED LABELS.
	preds = preds.split(' ')
	preds_labels = np.array(preds[preds_indices['label']::len(preds_indices)])
	preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)
	preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)
	preds_confidences = np.array(preds[preds_indices['confidence']::len(preds_indices)]).astype(float)

	# CHECK IF EACH PRED WAS EXTRANEOUS.
	for pred_index in range(len(preds_x)):
		ACCEPTABLE_MARGIN_PX = 5
		relevant_char_mask = (
			(preds_x[pred_index] > truth_xmins - ACCEPTABLE_MARGIN_PX) &
			(preds_x[pred_index] < truth_xmaxes + ACCEPTABLE_MARGIN_PX) &
			(preds_y[pred_index] > truth_ymins - ACCEPTABLE_MARGIN_PX) &
			(preds_y[pred_index] < truth_ymaxes + ACCEPTABLE_MARGIN_PX))

		pred_is_extraneous = sum(relevant_char_mask) == 0
		pred_is_correct = preds_labels[pred_index] in truth_labels[relevant_char_mask]
		confidence = preds_confidences[pred_index]
		if pred_is_extraneous:
			extraneous_pred_confidences.append(confidence)
		elif not pred_is_correct:
			incorrect_pred_confidences.append(confidence)
		else:
			correct_pred_confidences.append(confidence)

	return correct_pred_confidences, incorrect_pred_confidences, extraneous_pred_confidences

# (
#	mean confidence of correctly labeled chars,
#	std confidence of correctly labeled chars,
#	mean confidence of incorrectly labeled chars,
#	std confidence of incorrectly labeled chars,
#	mean confidence of extraneous chars,
#	std confidence of extraneous chars
# )
def GetConfidencesForAllImages(submission_file_contents, ground_truth_file_contents):
	all_correct_pred_confidences = []
	all_incorrect_pred_confidences = []
	all_extraneous_pred_confidences = []
	for image_id in submission_file_contents['image_id'].values:
		correct_pred_confidences, incorrect_pred_confidences, extraneous_pred_confidences = GetConfidencesByTypeForImage(
			submission_file_contents[submission_file_contents['image_id'] == image_id]['labels'].values[0],
			ground_truth_file_contents[ground_truth_file_contents['image_id'] == image_id]['labels'].values[0])
		all_correct_pred_confidences += correct_pred_confidences
		all_incorrect_pred_confidences += incorrect_pred_confidences
		all_extraneous_pred_confidences += extraneous_pred_confidences

	return (
		np.mean(all_correct_pred_confidences),
		np.std(all_correct_pred_confidences),
		np.mean(all_incorrect_pred_confidences),
		np.std(all_incorrect_pred_confidences),
		np.mean(all_extraneous_pred_confidences),
		np.std(all_extraneous_pred_confidences))

if __name__ == '__main__':
	# LOAD DATA.
	# submission_file_contents = pd.read_csv('RawSubmissionFiles/CrossValidationSetPredictions_75thresh.csv')
	submission_file_contents = pd.read_csv('RawSubmissionFiles/WithConfidence/CrossValidationSetPredictions_SubsetModelWithConfidence.csv')
	submission_image_ids = submission_file_contents['image_id'].values
	ground_truth_file_contents = pd.read_csv('../Datasets/GroundTruthBoundingBoxes.csv')
	sorted_charset = np.array(pickle.load(open('../Datasets/SortedCharset.p', 'rb')))
	
	calculate_confidence_stats = 'confidence' in preds_indices.keys()
	if calculate_confidence_stats:
		confidence_stats = GetConfidencesForAllImages(submission_file_contents, ground_truth_file_contents)

	# GET THE CONFUSION MATRIX.
	WORKER_COUNT = 8
	all_images_confusion_matrix = GenerateConfusionMatrixForImages(
		submission_file_contents, ground_truth_file_contents, sorted_charset, submission_image_ids, WORKER_COUNT)

	# GET STATS ABOUT WHERE THE LABELS ARE LOCATED RELATIVE TO THE GROUND TRUTH.
	mean_x_offset, std_x_offset, mean_y_offset, std_y_offset = GetAllImagesXYDistanceStats(
		submission_file_contents, 
		ground_truth_file_contents)

	# GET STATS ABOUT THE EXTRANEOUS CHARS THAT ARE DETECTED.
	p_extr_char, mu_extr_char, std_extr_char, p_extr_char_classes = GetAllImagesExtraneousCharStats(
		submission_file_contents, ground_truth_file_contents, sorted_charset)

	# GET STATS ABOUT HOW FREQUENTLY EACH TYPE OF CHARACTER FAILES TO BE DETECTED.
	failed_detection_probs = GetAllImagesFailedDetectionProbs(
		submission_file_contents, ground_truth_file_contents, sorted_charset)

	# DUMP THE STATISTICS TO A FILE.
	error_stats = {
		'ConfusionMatrix': all_images_confusion_matrix,
		'MeanXOffset': mean_x_offset,
		'MeanYOffset': mean_y_offset,
		'StdXOffset':std_x_offset,
		'StdYOffset':std_y_offset,
		'ProbImageContainsExtraneousChar': p_extr_char,
		'MeanImageExtraneousCharCount': mu_extr_char,
		'StdImageExtraneousCharCount': std_extr_char,
		'ExtraneousCharClassProbs': p_extr_char_classes,
		'FailedDetectionProbByCharClass': failed_detection_probs, 
	}
	if calculate_confidence_stats:
		error_stats['MeanCorrectLabelConfidence'] = confidence_stats[0]
		error_stats['StdCorrectLabelConfidence'] = confidence_stats[1]
		error_stats['MeanIncorrectLabelConfidence'] = confidence_stats[2]
		error_stats['stdIncorrectLabelConfidence'] = confidence_stats[3]
		error_stats['MeanExtraneousLabelConfidence'] = confidence_stats[4]
		error_stats['StdExtraneousLabelConfidence'] = confidence_stats[5]
	pickle.dump(error_stats, open('../Datasets/ErrorStats.p', 'wb'))