import sys
import random
import pandas as pd
import re
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def get_train_val_split(infile, seed = 1):
    folds = {}
    random.seed(seed)
    # create a dictionary partition
    # partition['train']: list of training IDs
    # partition['validation']: list of validation IDs
    length = len(pd.read_csv(infile))
    indices = list(range(length))
    random.shuffle(indices)
    one_fold = length // 5
    for i in range(1, 6):
        curr_dict = {}
        curr_dict['validation'] = indices[one_fold * (i - 1) : one_fold * i]
        curr_dict['train'] = list(set(indices) - set(curr_dict['validation']))
        folds[i] = curr_dict
    return folds

vocab = {}
def build_vocab(infile, fold, vocab_size = 1000, all=False):
    global vocab
    vocab = {}
    freq_dict = {}
    df = pd.read_csv(infile)
    if not all:
        # if do not use the entire dataset to build vocab
        df = df.iloc[fold['train'],:]
    # concat all tweets into one string
    tweets_string = ' '.join(df['Tweet text'])
    # remove punctuation, convert to lower case
    processed_tweets_string = re.sub(r'[^\w\s]', ' ', tweets_string).lower()
    # split into tokens
    tokens = processed_tweets_string.split()
    # create count dictionary freq_dict
    for token in tokens:
        if token in freq_dict:
            freq_dict[token] += 1
        else:
            freq_dict[token] = 1
    # sort dictionary in descending freq count
    sorted_freq = sorted(freq_dict.items(), key = lambda x:x[1], reverse = True)
    # take top vocab_size - 1 vocab, accounting for [UNK] token
    pruned_vocab = sorted_freq[:vocab_size - 1]
    # add to vocab
    vocab['[UNK]'] = 0
    for i, token_tuple in enumerate(pruned_vocab):
        vocab[token_tuple[0]] = i + 1
    return

vectors = {}
labels = {}
def vectorizer(infile, vocab_size = 1000):
    global vectors, labels
    vectors = {}
    labels = {}
    df = pd.read_csv(infile)
    for index, row in df.iterrows():
        vec = [0] * vocab_size
        tokens = list(set(re.sub(r'[^\w\s]', ' ', row['Tweet text']).lower().split()))
        for token in tokens:
            if token in vocab:
                vec[vocab[token]] = 1
            else:
                vec[0] = 1
        vectors[index] = vec
        labels[index] = int(row['label'])
    return (vectors, labels)


class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # generates one sample of data
        ID = self.list_IDs[index]
        X = torch.tensor(vectors[ID])
        y = self.labels[ID]
        return X, y


''' Model '''
class IronyClassifier(nn.Module):
    ''' A 2-layer Multilayer Perceptron for classifying tweets '''

    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        Args:
            input_dim (int): size of input vector(size of vocab)
            hidden_dim (int): output size of the first linear layer
            output_dim (int): output size of the second linear layer(number of classes)
        '''
        super(IronyClassifier, self).__init__()
        # fully connected layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    ''' Forward propogation of the classifier '''

    def forward(self, x_in, apply_softmax=False):
        '''
        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for softmax activation
                should be false if using Cross Entropy Loss
        Returns:
            result tensor. tensor.shape should be (batch, output_dim)
        '''
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices)

def run_training_loop(config):
    # 5-fold cross validation
    # get the indices for train and validation for each fold
    global vectors, labels
    folds = get_train_val_split(config['data_file'])
    partition = folds[config['fold']]
    # create vocab
    build_vocab(config['data_file'], partition, vocab_size=config['vocab_size'])
    assert len(vocab) == config['vocab_size']
    # create vectors and labels
    vectors, labels = vectorizer(config['data_file'], vocab_size=config['vocab_size'])
    # training loop
    training_set = Dataset(partition['train'], labels)
    training_generator = data.DataLoader(training_set, batch_size=config['batch_size'], shuffle=True)

    validation_set = Dataset(partition['validation'], labels)
    validation_generator = data.DataLoader(validation_set, batch_size=config['batch_size'], shuffle=True)

    if config['task'] == 'taskA':
        output_dim = 2
    elif config['task'] == 'taskB':
        output_dim = 4

    device = 'cpu'
    classifier = IronyClassifier(input_dim=config['vocab_size'], hidden_dim=config['hidden_dim'], output_dim=output_dim)
    classifier = classifier.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        # training
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
        batch_index = 0
        for local_batch, local_labels in training_generator:
            # clear gradients
            optimizer.zero_grad()
            # compute output
            y_pred = classifier(local_batch.float())
            # compute loss
            loss = loss_func(y_pred, local_labels)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # produce gradients
            loss.backward()
            # backpropogation
            optimizer.step()
            # compute accuracy
            acc_t = compute_accuracy(y_pred, local_labels)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            batch_index += 1
        #print("Epoch {}: ".format(epoch + 1))
    #print("Train Loss: {}".format(running_loss))
    print("Train Accuracy: {}".format(running_acc))

    # evaluation
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()
    batch_index = 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # get prediction
            y_pred = classifier(local_batch.float())
            loss = loss_func(y_pred, local_labels)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # compute accuracy
            acc_t = compute_accuracy(y_pred, local_labels)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            batch_index += 1
        #print("Val Loss: {}".format(running_loss))
        print("Val Accuracy: {}".format(running_acc))
    return

def run_test(config):
    build_vocab(config['train_file'], -1, config['vocab_size'], all=True)
    global vectors, labels
    partition = {}
    train_num = len(pd.read_csv(config['train_file']))
    val_num = len(pd.read_csv(config['test_file']))
    partition['train'] = list(range(train_num))
    partition['validation'] = list(range(train_num, train_num + val_num))

    # create vectors and labels
    train_vectors, train_labels = vectorizer(config['train_file'], vocab_size=config['vocab_size'])
    test_vectors, test_labels = vectorizer(config['test_file'], vocab_size=config['vocab_size'])
    test_vectors = {k+len(train_vectors) : v for k, v in test_vectors.items()}
    test_labels = {k+len(train_vectors): v for k, v in test_labels.items()}
    train_vectors.update(test_vectors)
    train_labels.update(test_labels)
    vectors = train_vectors
    labels = train_labels

    # training loop
    training_set = Dataset(partition['train'], labels)
    training_generator = data.DataLoader(training_set, batch_size=config['batch_size'], shuffle=True)

    validation_set = Dataset(partition['validation'], labels)
    validation_generator = data.DataLoader(validation_set, batch_size=config['batch_size'], shuffle=True)

    if config['task'] == 'taskA':
        output_dim = 2
    elif config['task'] == 'taskB':
        output_dim = 4

    device = 'cpu'
    classifier = IronyClassifier(input_dim=config['vocab_size'], hidden_dim=config['hidden_dim'], output_dim=output_dim)
    classifier = classifier.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config['learning_rate'])

    for epoch in range(config['num_epochs']):
        # training
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()
        batch_index = 0
        for local_batch, local_labels in training_generator:
            # clear gradients
            optimizer.zero_grad()
            # compute output
            y_pred = classifier(local_batch.float())
            # compute loss
            loss = loss_func(y_pred, local_labels)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # produce gradients
            loss.backward()
            # backpropogation
            optimizer.step()
            # compute accuracy
            acc_t = compute_accuracy(y_pred, local_labels)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            batch_index += 1
        # print("Epoch {}: ".format(epoch + 1))
    #print("Train Loss: {}".format(running_loss))
    print("Train Accuracy: {}".format(running_acc))

    # evaluation
    running_loss = 0.0
    running_acc = 0.0
    classifier.eval()
    batch_index = 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # get prediction
            y_pred = classifier(local_batch.float())
            loss = loss_func(y_pred, local_labels)
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # compute accuracy
            acc_t = compute_accuracy(y_pred, local_labels)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
            batch_index += 1
        #print("Test Loss: {}".format(running_loss))
        print("Test Accuracy: {}".format(running_acc))
    return

def plot_accuracy(x, y, z, title, xlabel, ylabel, filename):
    fig = plt.figure()
    test, = plt.plot(x, y, marker = 'o', color = 'royalblue', linestyle = '--')
    train, = plt.plot(x, z, marker = 'o', color = 'coral', linestyle = '--')
    plt.legend([test, train], ["test", "train"])
    plt.grid(color='mediumpurple', linestyle='--', alpha = 0.5)
    fig.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.show()
    plt.savefig('./plots/{}'.format(filename))
    return

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[0] == 'test':
        # python nn.py test taskA
        # test mode
        task = arguments[1]
        config = {}
        if task == 'taskA':
            # best set of parameters found using cross validation for taskA
            config['task'] = 'taskA'
            config['vocab_size'] = 100
            config['num_epochs'] = 10
            config['hidden_dim'] = 4
            config['learning_rate'] = 0.005
            config['batch_size'] = 32
            config['train_file'] = "../data/taskA/train_taskA.csv"
            config['test_file'] = "../data/taskA/test_taskA.csv"
        elif task == 'taskB':
            # best set of parameters found using cross validation for taskB
            config['task'] = 'taskB'
            config['vocab_size'] = 1200
            config['num_epochs'] = 4
            config['hidden_dim'] = 2
            config['learning_rate'] = 0.003
            config['batch_size'] = 8
            config['train_file'] = "../data/taskB/train_taskB.csv"
            config['test_file'] = "../data/taskB/test_taskB.csv"
        run_test(config)
    else:
        # python nn.py task fold vocab_size num_epochs hidden_dim learning_rate batch_size
        # python nn.py taskB 1 100 20 3 0.005 32
        config = {}
        config ['task'] = arguments[0]
        config['fold'] = int(arguments[1])
        config['vocab_size'] = int(arguments[2])
        config['num_epochs'] = int(arguments[3])
        config['hidden_dim'] = int(arguments[4])
        config['learning_rate'] = float(arguments[5])
        config['batch_size'] = int(arguments[6])
        if config['task'] == 'taskA':
            config['data_file'] = "../data/taskA/train_taskA.csv"
        elif config['task'] == 'taskB':
            config['data_file'] = "../data/taskB/train_taskB.csv"
        run_training_loop(config)
