labels_lookup = {}
lines = open('CorrectedSubmissionFile_debug.csv').readlines()
for line in lines:
	if line.split(',')[0] not in labels_lookup.keys():
		labels_lookup[line.split(',')[0]] = line.split(',')[1]
	if labels_lookup[line.split(',')[0]] != line.split(',')[1]:
		print('YEET!')