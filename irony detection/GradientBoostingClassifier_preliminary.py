import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# function for cross validation, return average accuracy across all folds
def cross_validation(train_df, classifier, folds):
	fold_size = int(len(train_df) / folds)
	low = 0					# lower bound index of validation set
	high = fold_size		# higher bound index of validation set
	accuracy_scores = []

	# cross validation
	for i in tqdm(range(folds)):
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type=str, default='train_taskA.csv', help='Train file')
	#parser.add_argument('--test', type=str, default='test_taskA.csv', help='Test file')
	args = parser.parse_args()

	print("")
	print("----- Gradient Boosting Classifier -----")

	# read from csv file
	train_df = pd.read_csv(args.train)
	#test_df = pd.read_csv(args.test)
	
	# shuffle datasets
	train_df = shuffle(train_df, random_state = 1)
	#test_df = shuffle(test_df, random_state = 1)

	# gradient boosting classifier
	gbc_clf = GradientBoostingClassifier(random_state = 1)
	cv_folds = 5 # cross validation folds
	cv_accuracy = cross_validation(train_df, gbc_clf, cv_folds)
	print("")
	print("Cross validation score:")
	print("Accuracy: {}".format(cv_accuracy))
	print("")
