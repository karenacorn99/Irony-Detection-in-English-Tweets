CURRENT_VOCAB_PATH = "../preprocessed/taskA/vocab.txt"
CURRENT_TWEET_PATH = "../preprocessed/taskA/train.csv"


def read_vocab(vocab_path):
    with open(vocab_path, encoding='utf8', errors='ignore') as f:
        vocab_ls = f.read().splitlines()
    return vocab_ls


def read_tweets(tweet_path, start_index, end_index):
    with open(tweet_path, encoding='utf8', errors='ignore') as f:
        tweets = [next(f) for _ in range(end_index + start_index)]
    return tweets[start_index:]


def parse_word(word, vocab):
    if not word:
        return []
    if word in vocab:
        return [word]
    if len(word) > 2:
        r_list = []
        left, right, rest = word[0], word[-1], word[1:-1]
        if rest in vocab:
            r_list.append(rest)
            if left in vocab:
                r_list.append(left)
            if right in vocab and right not in r_list:
                r_list.append(right)
        if r_list:
            return r_list
    print("ERROR: unable to match " + word)
    return []


def process_tweet_to_vector(tweet, input_vocab):
    delim = tweet.split(',')
    index, label = delim[0], delim[1]
    data = tweet.replace(index + ',' + label + ',', '', 1)
    raw_data = data.replace('\n', '').replace('\"', ' \" ').split(' ')
    raw_data = [word.replace(' ', '') for word in raw_data]
    parsed_words = []
    for word in raw_data:
        parsed_words += parse_word(word, input_vocab)
    return_vector = []
    for word in input_vocab:
        if word in parsed_words:
            return_vector.append(1)
        else:
            return_vector.append(0)
    return index, label, return_vector


def create_training_matrices(vocab_path, tweet_path, start_index, end_index):
    X_build, y_build = [], []
    my_vocab = read_vocab(vocab_path)
    my_tweets = read_tweets(tweet_path, start_index, end_index)
    for i in range(end_index - start_index):
        _, label, vector = process_tweet_to_vector(my_tweets[i], my_vocab)
        y_build.append(label)
        X_build.append(vector)
    return X_build, y_build


if __name__ == "__main__":
    print("Testing creation of training matrices X and y")
    X, y = create_training_matrices(CURRENT_VOCAB_PATH, CURRENT_TWEET_PATH, 1, 3001)
    # print(X[1])
    # print(y[1])
    print("Done with test")

