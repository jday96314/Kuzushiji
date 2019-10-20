with open('train.csv') as training_file:
	training_lines = training_file.readlines()

with open('every_fourth_cross_validation.csv', 'w') as output_file:
	output_file.write(training_lines[0])
	for index, line in enumerate(training_lines):
		if (index - 1)%4 == 0:
			output_file.write(line)