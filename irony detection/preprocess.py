import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import demoji
#demoji.download_codes() # uncomment if download needed
from nltk.tokenize import TweetTokenizer
from argparse import Namespace
from collections import Counter
import json
import os
import string
import re

'''
All urls will take the [URL] token
All usertag will take the [USER] token
All emojis will be translated to text surrounded by :
    examples, 💯 will be :hundred points:
alphanumeric and puntuations will be left as they are
'''
# preprocess function that make the above adjustments to tweet text
def preprocess(text):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    emojis = demoji.findall(text)
    cleaned = []
    for token in tokens:
        if 'http' in token:
            cleaned.append('[URL]')
        elif '@' in token:
            cleaned.append('[USER]')
        elif token in emojis:
            cleaned.append(':' + ''.join(emojis[token].split()) + ':')
        else:
            cleaned.append(token.lower())
    return ' '.join(cleaned)

# function to display the label distributions
def count_labels(y):
    len_y = len(y)
    ctr = Counter(y)
    ctr_items = ctr.items()
    for key, val in ctr_items:
        print('{}: {}'.format(key,  val))
    print('')

# function to drop some entries and realign indexes
def drop_entries(df, idxs):
    new_df = df.drop(df.index[idxs])
    new_len = len(new_df)
    tweet_idx = [i + 1 for i in range(new_len)]
    new_df['Tweet index'] = tweet_idx
    return new_df

# funtion to split into train and test sets and realign indexes
def train_test_split(df, msg):
    train_df = df.iloc[0:3000]
    test_df = df.iloc[3000:3817]
    test_idxs = [i + 1 for i in range(817)]
    test_df['Tweet index'] = test_idxs
    train_df.to_csv('train_task{}.csv'.format(msg), index=False)
    test_df.to_csv('test_task{}.csv'.format(msg), index=False)

# original file names
taskA_file = 'train_emoji_A.txt'
taskB_file = 'train_emoji_B.txt'

# parse original text files
with open(taskA_file, 'r') as f:
    taskA_lines = [line.split('\t') for line in f.readlines()][1:]
assert len(taskA_lines) == 3834
with open(taskB_file, 'r') as f:
    taskB_lines = [line.split('\t') for line in f.readlines()][1:]
assert len(taskB_lines) == 3834

# put parsed lines into dataframes
taskA_full = pd.DataFrame(taskA_lines, columns = ['Tweet index', 'label', 'Tweet text'])
taskB_full = pd.DataFrame(taskB_lines, columns = ['Tweet index', 'label', 'Tweet text'])

# clean tweet texts
cleaned_A = taskA_full['Tweet text'].map(preprocess)
cleaned_B = taskB_full['Tweet text'].map(preprocess)
taskA_full['Tweet text'] = cleaned_A
taskB_full['Tweet text'] = cleaned_B

# drop some entries to have the label distribution described in the project plan
idxs = [14,94,601,1042,1574,1643,2186,270,282,474,1067,1289,1833,2352,2483,2784,1353]
idxs = [i - 1 for i in idxs]
taskA_full = drop_entries(taskA_full, idxs)
print('Task A label distribution')
count_labels(taskA_full['label'].values)
print('')
taskB_full = drop_entries(taskB_full, idxs)
print('Task B label distribution')
count_labels(taskB_full['label'].values)

# split the tweets into train and test sets
train_test_split(taskA_full, 'A')
train_test_split(taskB_full, 'B')



