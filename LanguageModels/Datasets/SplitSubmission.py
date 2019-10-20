with open('ObjectDetectorPredictions.csv') as bounding_boxes_file:
	lines = bounding_boxes_file.read().split('\n')

with open('TrainingObjectDetectorPredictions.csv', 'w') as output_training_file:
	with open('TestingObjectDetectorPredictions.csv', 'w') as output_testing_file:
		output_training_file.write(lines[0] + '\n')
		output_training_file.writelines([line + "\n" for line in lines[1:len(lines)//2]])

		output_testing_file.write(lines[0] + '\n')
		output_testing_file.writelines([line + "\n" for line in lines[len(lines)//2:]])

		# for line_index, line in enumerate(lines):
		# 	if len(line) == 0:
		# 		break

		# 	if line_index == 0:
		# 		output_training_file.write(line + '\n')
		# 		output_testing_file.write(line + '\n')
		# 	elif line_index%2 == 0:
		# 		output_training_file.write(line + "\n")
		# 	else:
		# 		output_testing_file.write(line + "\n")
