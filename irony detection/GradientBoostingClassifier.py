import sys
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier

# function to calculate accuracy
def accuracy_score(y_true, y_pred):
	y_true_np = y_true.to_numpy()
	correct_count = 0
	for i in range(len(y_true)):
		if y_true_np[i] == y_pred[i]:
			correct_count += 1
	return correct_count / len(y_true)

# function for cross validation, return average accuracy across all folds
def cross_validation(train_df, classifier, folds):
	fold_size = int(len(train_df) / folds)
	low = 0					# lower bound index of validation set
	high = fold_size		# higher bound index of validation set
	accuracy_scores = []

	# cross validation
	for i in range(folds):
		# cross validation train set and validation set
		cv_valid_df = train_df.iloc[low:high]
		cv_train_df = pd.concat([train_df, cv_valid_df]).drop_duplicates(keep = False)

		# count vectorizer to transform train and validation sets
		vectorizer = CountVectorizer()
		vectorizer.fit(cv_train_df['Tweet text'])
		cv_train_X = vectorizer.transform(cv_train_df['Tweet text'])
		cv_valid_X = vectorizer.transform(cv_valid_df['Tweet text'])

		# train classifier and make predictions on validation set
		classifier.fit(cv_train_X, cv_train_df['label'])
		y_pred = classifier.predict(cv_valid_X)

		# validation set prediction scores
		accuracy = accuracy_score(cv_valid_df['label'], y_pred)
		accuracy_scores.append(accuracy)

		# go to next fold
		low += fold_size
		high += fold_size

	# average across all folds
	return (sum(accuracy_scores) / len(accuracy_scores))

# function to get accuracy score of test set
def test_set_accuracy(train_df, test_df, classifier):
	vectorizer = CountVectorizer()
	vectorizer.fit(train_df['Tweet text'])
	train_X = vectorizer.transform(train_df['Tweet text'])
	test_X = vectorizer.transform(test_df['Tweet text'])
	classifier.fit(train_X, train_df['label'])
	y_pred = classifier.predict(test_X)
	accuracy = accuracy_score(test_df['label'], y_pred)
	return accuracy

# function to plot in 2D
def plot_2D(xs, ys, message):
	plt.plot(xs, ys)
	plt.title(message)
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--trainA', type=str, default='train_taskA.csv', help='Task A Train file')
	parser.add_argument('--testA', type=str, default='test_taskA.csv', help='Task A Test file')
	parser.add_argument('--trainB', type=str, default='train_taskB.csv', help='Task B Train file')
	parser.add_argument('--testB', type=str, default='test_taskB.csv', help='Task B Test file')
	parser.add_argument('--task', type=str, default='A', help='Task A or B')
	parser.add_argument('--option', type=int, default=5, help='Parameter(s) to tune (see manual.txt)')
	args = parser.parse_args()

	print("")
	print("----- Gradient Boosting Classifier -----")

	folds = 5
	# read from csv file
	train_df = pd.read_csv(args.trainA)
	test_df = pd.read_csv(args.testA)
	if (args.task == 'B'):
		train_df = pd.read_csv(args.trainB)
		test_df = pd.read_csv(args.testB)
	
	# learning_rate tuning
	if (args.option == 1):
		print("")
		print("--- learning_rate tuning ---")
		top_score = 0
		top_parameter = -1
		parameter_range = np.arange(0.1, 1.01, 0.1)
		scores = []
		for parameter in tqdm(parameter_range):
			classifier = GradientBoostingClassifier(learning_rate = parameter, random_state = 1)
			accuracy = cross_validation(train_df, classifier, folds)
			scores.append(accuracy)
			if accuracy > top_score:
				top_score = accuracy
				top_parameter = parameter
		print('Top score: {}'.format(top_score))
		print('Top learning_rate: {}'.format(top_parameter))
		plot_2D(parameter_range, scores, 'learning_rate')

	# n_estimators tuning
	if (args.option == 2):
		print("")
		print("--- n_estimators tuning ---")
		top_score = 0
		top_parameter = -1
		parameter_range = range(100, 1200, 50)
		scores = []
		for parameter in tqdm(parameter_range):
			classifier = GradientBoostingClassifier(n_estimators = parameter, random_state = 1)
			accuracy = cross_validation(train_df, classifier, folds)
			scores.append(accuracy)
			if accuracy > top_score:
				top_score = accuracy
				top_parameter = parameter
		print('Top score: {}'.format(top_score))
		print('Top n_estimator: {}'.format(top_parameter))
		plot_2D(parameter_range, scores, 'n_estimator')

	# min_samples_split tuning
	if (args.option == 3):
		print("")
		print("--- min_samples_split tuning ---")
		top_score = 0
		top_parameter = -1
		parameter_range = range(2, 15, 1)
		scores = []
		for parameter in tqdm(parameter_range):
			classifier = GradientBoostingClassifier(min_samples_split = parameter, random_state = 1)
			accuracy = cross_validation(train_df, classifier, folds)
			scores.append(accuracy)
			if accuracy > top_score:
				top_score = accuracy
				top_parameter = parameter
		print('Top score: {}'.format(top_score))
		print('Top min_samples_split: {}'.format(top_parameter))
		plot_2D(parameter_range, scores, 'min_samples_split')

	# max_depth tuning
	if (args.option == 4):
		print("")
		print("--- max_depth tuning ---")
		top_score = 0
		top_parameter = -1
		parameter_range = np.arange(2, 15, 1)
		scores = []
		for parameter in tqdm(parameter_range):
			classifier = GradientBoostingClassifier(max_depth = parameter, random_state = 1)
			accuracy = cross_validation(train_df, classifier, folds)
			scores.append(accuracy)
			if accuracy > top_score:
				top_score = accuracy
				top_parameter = parameter
		print('Top score: {}'.format(top_score))
		print('Top max_depth: {}'.format(top_parameter))
		plot_2D(parameter_range, scores, 'max_depth')

	# full tuning procedure
	if (args.option == 5):
		print('')
		print('--- full tuning procedure ---')

		# ranges of the parameters
		lr_range = np.arange(0.1, 0.71, 0.1)
		n_est_range = range(700, 1010, 50)
		mss_range = range(7, 12, 1)
		md_range = range(2, 9, 1)
		if (args.task == 'B'):
			lr_range = np.arange(0.1, 0.41, 0.1)
			n_est_range = range(300, 550, 50)
			mss_range = list(range(2, 8, 1))+list(range(12, 15, 1))
			md_range = range(3, 11, 1)

		# max_depth and min_samples_split tuning
		print('')
		print('--- tuning min_samples_split and max_depth ---')
		top_score = 0
		top_mss_md = np.zeros([2])
		for mss in tqdm(mss_range):
			for md in tqdm(md_range):
				classifier = GradientBoostingClassifier(min_samples_split = mss, max_depth = md, random_state = 1)
				accuracy = cross_validation(train_df, classifier, folds)
				if accuracy > top_score:
					top_score = accuracy
					top_mss_md[0] = int(mss)
					top_mss_md[1] = int(md)
		print('Top min_samples_split: {}'.format(top_mss_md[0]))
		print('Top max_depth: {}'.format(top_mss_md[1]))	

		# learning_rate and n_estimator tuning	
		print('')
		print('--- tuning learning_rate and n_estimator ---')
		top_score = 0
		top_lr_nest = np.zeros([2])
		for lr in tqdm(lr_range):
			for n_est in tqdm(n_est_range):
				classifier = GradientBoostingClassifier(min_samples_split = int(top_mss_md[0]), max_depth = int(top_mss_md[1]), learning_rate = lr, n_estimators = n_est, random_state = 1)
				accuracy = cross_validation(train_df, classifier, folds)
				if accuracy > top_score:
					top_score = accuracy
					top_lr_nest[0] = lr
					top_lr_nest[1] = int(n_est)
		print('Top learning_rate: {}'.format(top_lr_nest[0]))
		print('Top n_estimators: {}'.format(top_lr_nest[1]))

		# calculate scores
		default_classifier = GradientBoostingClassifier(random_state = 1)
		tuned_classifier = GradientBoostingClassifier(min_samples_split = int(top_mss_md[0]), max_depth = int(top_mss_md[1]), learning_rate = top_lr_nest[0], n_estimators = int(top_lr_nest[1]), random_state = 1)
		default_accuracy = cross_validation(train_df, default_classifier, folds)
		tuned_accuracy = cross_validation(train_df, tuned_classifier, folds)
		tuned_test_accuracy = test_set_accuracy(train_df, test_df, tuned_classifier)

		# print results
		print('')
		print('Default cross validation score: {}'.format(default_accuracy))
		print('Tuned cross validation score: {}'.format(tuned_accuracy))
		print('Test set score: {}'.format(tuned_test_accuracy))
		print('')



