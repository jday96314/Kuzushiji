import torch
import math

def ComputeIntersectOverSmallest(center_x_1, center_y_1, w1, h1, center_x_2, center_y_2, w2, h2):
	if abs(center_x_1 - center_x_2) > (w1//2 + w2//2):
		return 0
	if abs(center_y_1 - center_y_2) > (h1//2 + h2//2):
		return 0

	x1 = max(center_x_1 - w1/2, center_x_2 - w2/2)
	y1 = max(center_y_1 - h1/2, center_y_2 - h2/2)
	x2 = min(center_x_1 + w1/2, center_x_2 + w2/2)
	y2 = min(center_y_1 + h1/2, center_y_2 + h2/2)

	shared_area = max((x2 - x1) * (y2 - y1), 0)
	smallest_box_area = min(w1*h1, w2*h2)

	return shared_area / smallest_box_area

def SuppressNonMaximalBboxes(
		anchors, 
		anchor_classes, 
		bbox_regression_outputs, 
		scaled_image_widths,
		scaled_image_heights,
		min_confidence_threshold = .6,
		min_suppression_ios = .3):
	# COMPUTE THE PROBABILITY OF EACH ANCHOR BEING A BOUNDING BOX.
	p_chars = torch.nn.functional.softmax(anchor_classes, dim = 0)[1].detach().cpu().numpy()

	# GET ALL THE BOUNDING BOXES.
	cpu_bbox_regression_outputs = bbox_regression_outputs.detach().cpu().numpy()
	bboxes_with_confidences = []
	anchor_class_count, anchor_count, fmap_width, fmap_height = anchor_classes.shape
	fmap_x_stride = scaled_image_widths // fmap_width
	fmap_y_stride = scaled_image_heights // fmap_height
	for anchor_index in range(len(anchors)):
		for fmap_x_index in range(fmap_width):
			for fmap_y_index in range(fmap_height):
				is_char_confidence = p_chars[anchor_index][fmap_x_index][fmap_y_index]
				if p_chars[anchor_index][fmap_x_index][fmap_y_index] > min_confidence_threshold:
					anchor_center_x = int((fmap_x_index + .5) * fmap_x_stride)
					anchor_center_y = int((fmap_y_index + .5) * fmap_y_stride)
					anchor_w, anchor_h = anchors[anchor_index]

					tx = cpu_bbox_regression_outputs[0][anchor_index][fmap_x_index][fmap_y_index]
					ty = cpu_bbox_regression_outputs[1][anchor_index][fmap_x_index][fmap_y_index]
					tw = cpu_bbox_regression_outputs[2][anchor_index][fmap_x_index][fmap_y_index]
					th = cpu_bbox_regression_outputs[3][anchor_index][fmap_x_index][fmap_y_index]

					bbox_center_x = anchor_center_x + anchor_w*tx
					bbox_center_y = anchor_center_y + anchor_h*ty
					bbox_w = math.exp(tw)*anchor_w
					bbox_h = math.exp(th)*anchor_h

					bboxes_with_confidences.append((
						bbox_center_x,
						bbox_center_y,
						bbox_w,
						bbox_h,
						is_char_confidence))

	# SORT THE BBOXES BY CONFIDENCE.
	# The most likely to be real characters are first.
	sorted_bboxes_with_confidence = sorted(
		bboxes_with_confidences,
		key = lambda bbox: bbox[-1],
		reverse = True)

	# FILTER THE BBOXES.
	maximal_bboxes = []
	for candidate_bbox_with_confidence in sorted_bboxes_with_confidence:
		# PARSE THE CANDIDATE BBOX.
		# We don't need the confidence any more.
		candidate_bbox_x, candidate_bbox_y, candidate_bbox_w, candidate_bbox_h, _ = candidate_bbox_with_confidence
		candidate_bbox = (candidate_bbox_x, candidate_bbox_y, candidate_bbox_w, candidate_bbox_h)

		# CHECK IF THE CANDIDATE BBOX OVERLAPS HEAVILY WITH ANY OF THE KNOWN BBOXES.
		candidate_overlaps_with_known_maxima = False
		for maximal_bbox in maximal_bboxes:
			ios = ComputeIntersectOverSmallest(*maximal_bbox, *candidate_bbox)
			if ios > min_suppression_ios:
				candidate_overlaps_with_known_maxima = True
				break
		if not candidate_overlaps_with_known_maxima:
			maximal_bboxes.append(candidate_bbox)

	return maximal_bboxes