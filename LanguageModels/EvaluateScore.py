"""
Python equivalent of the Kuzushiji competition metric (https://www.kaggle.com/c/kuzushiji-recognition/)
Kaggle's backend uses a C# implementation of the same metric. This version is
provided for convenience only; in the event of any discrepancies the C# implementation
is the master version.

Tested on Python 3.6 with numpy 1.16.4 and pandas 0.24.2.
"""


import argparse
import multiprocessing

import numpy as np
import pandas as pd


def define_console_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--sub_path', type=str)
	parser.add_argument('--solution_path', type=str)
	return parser


def score_page(preds, truth):
	"""
	Scores a single page.
	Args:
		preds: prediction string of labels and center points.
		truth: ground truth string of labels and bounding boxes.
	Returns:
		True/false positive and false negative counts for the page
	"""
	tp = 0
	fp = 0
	fn = 0

	truth_indices = {
		'label': 0,
		'X': 1,
		'Y': 2,
		'Width': 3,
		'Height': 4
	}
	preds_indices = {
		'label': 0,
		'X': 1,
		'Y': 2
	}

	if pd.isna(truth) and pd.isna(preds):
		return {'tp': tp, 'fp': fp, 'fn': fn, 'correct_label_slightly_outside_bbox_count' : 0, 'wrong_label_count' : 0}

	if pd.isna(truth):
		fp += len(preds.split(' ')) // len(preds_indices)
		return {'tp': tp, 'fp': fp, 'fn': fn, 'correct_label_slightly_outside_bbox_count' : 0, 'wrong_label_count' : 0}

	if pd.isna(preds):
		fn += len(truth.split(' ')) // len(truth_indices)
		return {'tp': tp, 'fp': fp, 'fn': fn, 'correct_label_slightly_outside_bbox_count' : 0, 'wrong_label_count' : 0}

	truth = truth.split(' ')
	if len(truth) % len(truth_indices) != 0:
		raise ValueError('Malformed solution string')
	truth_label = np.array(truth[truth_indices['label']::len(truth_indices)])
	truth_xmin = np.array(truth[truth_indices['X']::len(truth_indices)]).astype(float)
	truth_ymin = np.array(truth[truth_indices['Y']::len(truth_indices)]).astype(float)
	truth_xmax = truth_xmin + np.array(truth[truth_indices['Width']::len(truth_indices)]).astype(float)
	truth_ymax = truth_ymin + np.array(truth[truth_indices['Height']::len(truth_indices)]).astype(float)

	preds = preds.split(' ')
	if len(preds) % len(preds_indices) != 0:
		raise ValueError('Malformed prediction string')
	preds_label = np.array(preds[preds_indices['label']::len(preds_indices)])
	preds_x = np.array(preds[preds_indices['X']::len(preds_indices)]).astype(float)# - 15
	preds_y = np.array(preds[preds_indices['Y']::len(preds_indices)]).astype(float)# - 25
	preds_unused = np.ones(len(preds_label)).astype(bool)

	wrong_label_count = 0
	correct_label_slightly_outside_bbox_count = 0
	for xmin, xmax, ymin, ymax, label in zip(truth_xmin, truth_xmax, truth_ymin, truth_ymax, truth_label):
		# Matching = point inside box & character same & prediction not already used
		matching_predictions_mask = ((xmin < preds_x) & (xmax > preds_x) & (ymin < preds_y) & (ymax > preds_y) & (preds_label == label) & preds_unused)
		matched_successfully = matching_predictions_mask.sum() > 0
		
		wrong_label_predictions_mask = (xmin < preds_x) & (xmax > preds_x) & (ymin < preds_y) & (ymax > preds_y) & (preds_label != label) & preds_unused
		wrong_label_count += (wrong_label_predictions_mask.sum() > 0) and not matched_successfully
		# if (wrong_label_predictions_mask.sum() > 0) > 0:
		#	 print('Real label: {}, Predicted: {}'.format(label, preds_label[np.argmax(wrong_label_predictions_mask)]))

		correct_label_slightly_outside_bbox = ((xmin - 10 < preds_x) & (xmax + 10 > preds_x) & (ymin - 10 < preds_y) & (ymax + 10 > preds_y) & (preds_label == label) & (1 - matching_predictions_mask) & preds_unused).sum() > 0
		correct_label_slightly_outside_bbox_count += correct_label_slightly_outside_bbox and not matched_successfully
		# if correct_label_slightly_outside_bbox_count:
		#	 print((xmin - 10 < preds_x) & (xmax + 10 > preds_x) & (ymin - 10 < preds_y) & (ymax + 10 > preds_y) & (preds_label == label) & (1 - matching_predictions_mask) & preds_unused)
		#	 misplaced_pred_index = np.argmax((xmin - 10 < preds_x) & (xmax + 10 > preds_x) & (ymin - 10 < preds_y) & (ymax + 10 > preds_y) & (preds_label == label) & (1 - matching_predictions_mask) & preds_unused)
		#	 print(misplaced_pred_index)
		#	 print('x range: {}, x: {}\t\ty range: {}, y: {}\t\tChar: {}'.format(
		#		 (xmin, xmax),
		#		 preds_x[misplaced_pred_index],
		#		 (ymin, ymax),
		#		 preds_y[misplaced_pred_index],
		#		 preds_label[misplaced_pred_index]))
		#	 exit()

		if not matched_successfully:
			fn += 1
		else:
			tp += 1
			preds_unused[np.argmax(matching_predictions_mask)] = False
	fp += preds_unused.sum()
	return {
		'tp': tp, 
		'fp': fp, 
		'fn': fn, 
		'correct_label_slightly_outside_bbox_count' : correct_label_slightly_outside_bbox_count,
		'wrong_label_count' : wrong_label_count}

def kuzushiji_f1(sub, solution):
	"""
	Calculates the competition metric.
	Args:
		sub: submissions, as a Pandas dataframe
		solution: solution, as a Pandas dataframe
	Returns:
		f1 score
	"""
	if len(sub['image_id'].values) != len(solution['image_id'].values):
		print('WARNING: Submission image count of {} did not match ground image count ({}).'.format(
			len(sub['image_id'].values),
			len(solution['image_id'].values)))
		solution = solution[solution['image_id'].isin(sub['image_id'].values)]

	if not all(sub['image_id'].values == solution['image_id'].values):
		print(list(sub['image_id'].values[:3]), list(sub['image_id'].values[-3:]))
		print(list(solution['image_id'].values[:3]), list(solution['image_id'].values[-3:]))
		raise ValueError("Submission image id codes don't match solution")

	# pool = multiprocessing.Pool()
	# results = pool.starmap(score_page, zip(sub['labels'].values, solution['labels'].values))
	# pool.close()
	# pool.join()
	results = [
		score_page(sub['labels'].values[i], solution['labels'].values[i])
		for i
		in range(len(sub['labels'].values))]

	tp = sum([x['tp'] for x in results])
	fp = sum([x['fp'] for x in results])
	fn = sum([x['fn'] for x in results])
	correct_label_slightly_outside_bbox_count = sum([x['correct_label_slightly_outside_bbox_count'] for x in results])
	wrong_label_count = sum([x['wrong_label_count'] for x in results])

	print('Successful matches (tp): {}'.format(tp))
	print('Predicted labels that didn\'t match (fp): {}'.format(fp))
	print('Chars that weren\'t successfully matched (fn): {}'.format(fn))

	if (tp + fp) == 0 or (tp + fn) == 0:
		return 0
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	print('Precision: {:.05f}'.format(precision))
	print('Recall: {:.05f}'.format(recall))
	print('Correct labels slightly outside bounding boxes: {}'.format(correct_label_slightly_outside_bbox_count))
	print('Wrong lables: {}'.format(wrong_label_count))

	if precision > 0 and recall > 0:
		f1 = (2 * precision * recall) / (precision + recall)
	else:
		f1 = 0
	return f1


if __name__ == '__main__':
	parser = define_console_parser()
	shell_args = parser.parse_args()
	sub = pd.read_csv(shell_args.sub_path)
	solution = pd.read_csv(shell_args.solution_path)
	sub = sub.rename(columns={'rowId': 'image_id', 'PredictionString': 'labels'})
	solution = solution.rename(columns={'rowId': 'image_id', 'PredictionString': 'labels'})
	score = kuzushiji_f1(sub, solution)
	print('F1 score of: {0}'.format(score))