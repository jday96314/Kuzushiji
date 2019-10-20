import subprocess

best_pair = (-1, -1)
best_score = -1
for revision_threshold in [0, .25, .5, .75, .9]:
	for addition_threshold in [0, .25, .5, .75, .9, 1]:
		print('Testing revision threshold {}, addition threshold {}'.format(
			revision_threshold, 
			addition_threshold))
		subprocess.run([
			"python", 
			"SubmissionFileCorrector.py", 
			str(revision_threshold), 
			str(addition_threshold)])
		corrector_output = subprocess.check_output([
			"python",
			"../EvaluateScore.py",
			"--sub_path", 
			"CorrectedSubmissionFile.csv",
			"--solution_path", 
			"../Datasets/GroundTruthBoundingBoxes.csv"])
		f1_score = float(str(corrector_output).split(': ')[-1][:-3])
		print('Score:', f1_score)
		
		if f1_score > best_score:
			best_score = f1_score
			best_pair = (revision_threshold, addition_threshold)

print('Best configuration - revision threshold {}, addition threshold {}'.format(*best_pair))