# The following statistics need to be retrieved:
# - A confusion matrix where confusion_matrix[ground_truth_ord][predicted_ord] = count
# - The mean and standard deviation of the distance between the predicted label and thje bounding box's
#	ground truth center.
# - The mean and standard deviatioin of the number of extraneous characters (per image)
# - The probability of each individual character failing to be matched (with some extra margin around the
#	bounding boxes)