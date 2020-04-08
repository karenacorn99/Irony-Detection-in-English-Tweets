import create_vectors as vectorize
from sklearn import svm
import numpy as np
import random


class Kernel:
    def __init__(self, name, func):
        self.func = func
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.name


def compute_gram_matrix(X1, X2, K_func):
    # Returns the gram matrix for the input function K_func
    n1 = len(X1)
    n2 = len(X2)
    K = [[0 for _ in range(n2)] for _ in range(n1)]
    print("computing the gram matrix for " + str(K_func))
    print("this may take a while...")
    for i in range(n1):
        for j in range(n2):
            K[i][j] = K_func(np.array(X1[i]), np.array(X2[j]))
    print("computed gram matrix")
    return K


def dot(Xi, Xj):
    return np.dot(Xi, Xj)


def cross_validate_svm(X, y, k_fold=5, input_kernel=None):
    # fits k svm machines on input training matrices X and y, returning average accuracy with k-fold cross validation
    # Can take a kernel as an argument which will then be used to fit model, if none specified, no kernel will be used

    # Shuffle the training data randomly
    n = len(X)
    X_y_paired = [(X[i], y[i]) for i in range(n)]
    random.shuffle(X_y_paired)
    X_shuffled = [pair[0] for pair in X_y_paired]
    y_shuffled = [pair[1] for pair in X_y_paired]

    # Begin cross validation
    acc_list = []
    for i in range(k_fold):
        break_1 = int(i*n/k_fold)
        break_2 = int((i+1)*n/k_fold)
        X_validate_set = X_shuffled[break_1:break_2]
        X_training_set = X_shuffled[:break_1] + X_shuffled[break_2:]
        y_validate_set = y_shuffled[break_1:break_2]
        y_training_set = y_shuffled[:break_1] + y_shuffled[break_2:]
        if input_kernel:
            # A kernel was given, compute gram matrix for SVM fitting an predicting
            clf = svm.SVC(kernel='precomputed')
            X_training_set_gram = compute_gram_matrix(X_training_set, X_training_set, input_kernel)
            X_validate_set = compute_gram_matrix(X_validate_set, X_training_set, input_kernel)
            X_training_set = X_training_set_gram
        else:
            clf = svm.SVC()

        # Fit and predict SVM
        clf.fit(X_training_set, y_training_set)
        y_pred = clf.predict(X_validate_set)
        acc = [1 for j in range(len(y_validate_set)) if y_pred[j] == y_validate_set[j]]
        acc_list.append(sum(acc)/len(y_validate_set))
        print("Accuracy on fold " + str(i+1) + ": " + str(sum(acc)/len(y_validate_set)))

    # Display results
    if input_kernel:
        kern_str = str(input_kernel)
    else:
        kern_str = "no kernel"
    print("Average accuracy for SVM with " + kern_str + ": " + str(sum(acc_list)/k_fold))
    return sum(acc_list)/k_fold


if __name__ == "__main__":
    # Modify numerical arguments of create_training_matrices to quickly test functionality, creating gram
    # matrices for full data sets is immensely time consuming (hours)
    X, y = vectorize.create_training_matrices(vectorize.CURRENT_VOCAB_PATH, vectorize.CURRENT_TWEET_PATH, 1, 100)
    linear_kernel = Kernel('linear kernel', dot)
    cross_validate_svm(X, y, k_fold=5, input_kernel=linear_kernel)



