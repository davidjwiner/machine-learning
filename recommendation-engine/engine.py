#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 23:10:55 2016

@author: davidjwiner
"""

import scipy.io as sio
import numpy as np
import sklearn.metrics as metrics
import csv

def load_joke_training(handleNans=False):
    data = sio.loadmat('./joke_data/joke_train.mat')
    X = data['train']
    if (handleNans):
        return np.nan_to_num(X)
    else:
        return X
    
def predict(U, V, d, pairs, MSE=False):
    num_pairs = pairs.shape[0]
    result = np.zeros(num_pairs)
        
    # users and jokes are 1-indexed
    
    for i, pair in enumerate(pairs):
        u_num = int(pair[0])
        j_num = int(pair[1])
        result[i] += np.dot(U[u_num - 1,:], V[:, j_num - 1])
    
    if MSE:
        return result
    else:
        return result > 0

def singular_value_decomposition(X, d):
    u, s_temp, v = np.linalg.svd(X, full_matrices = False)
    s = np.diag(s_temp)
    u_temp = u[:, :d]
    v_temp = v[:d, :]
    s_temp = s[:d, :d]
    
    s_root = np.sqrt(s_temp)
    U = np.dot(u_temp, s_root)
    V = np.dot(s_root, v_temp)
    
    return U, V

def matrix_completion(X, d, lam):
    n = X.shape[0] # users
    k = X.shape[1] # jokes
    
    U = np.random.rand(n, d)
    V = np.random.rand(d, k)

    loss = 1e6
    last_loss = 0
    t = 0
    
    epsilons = {
                2: 0.1,
                5: 0.1,
                10: 1,
                20: 3
    }
    
    eps = epsilons[d]
    
    while np.abs(loss - last_loss) > eps:
        t += 1
        last_loss = loss
        for i in np.arange(n):
            nans = np.isnan(X[i, :])
            R_i = X[i, nans==False] 
            # nans is 1 x k matrix
            V_subset = V[:, nans == False]
            to_inv = np.dot(V_subset, V_subset.T) + lam * np.eye(d)
            inverse = np.linalg.solve(to_inv, np.eye(d))
            temp = np.dot(V_subset, R_i.T)
            U[i,:] = np.dot(inverse, temp)
            
        for j in np.arange(k):
            nans = np.isnan(X[:, j])
            R_j = X[nans==False, j]
            # nans is 1 x n matrix
            U_subset = U[nans == False, :]
            to_inv = np.dot(U_subset.T, U_subset) + lam * np.eye(d)
            inverse = np.linalg.solve(to_inv, np.eye(d))
            temp = np.dot(U_subset.T, R_j)
            V[:,j] = np.dot(inverse, temp)
        
        X_hat = np.dot(U, V)
        nans = np.isnan(X)
        loss = np.linalg.norm(X[nans == False] - X_hat[nans == False])
        print("Iteration {0} and the loss is {1}".format(t, loss))
    
    return U, V

def mean_squared_error(prediction, true_labels):
    n = prediction.size
    errors = prediction - true_labels
    return np.sum((errors * errors)) / n
    
# Loading data

X_replace_nans = load_joke_training(True)
X = load_joke_training(False)
X_valid = np.loadtxt('./joke_data/validation.txt', delimiter=',')
X_test = np.loadtxt('./joke_data/query.txt', delimiter=',')

true_pairs = np.zeros((X.shape[0]*100, 3))
for i in range(X.shape[0]):
    for j in range(100):
        if np.isnan(X[i,j]):
            continue
        else:
            user = i + 1
            joke = j + 1
            true_pairs[((100) * i) + j, :] = [user, joke, X[i, j]]

np.save('joke_train_pairs.npy', true_pairs)

training_pairs = np.load('joke_train_pairs.npy')[:, :2]
training_labels = np.load('joke_train_pairs.npy')[:, 2]
valid_pairs = X_valid[:, :2]
valid_labels = X_valid[:, 2]
test_pairs = X_valid[:, 1:]

# SVD
for d in [2, 5, 10, 20]:
    print("d = {0}".format(d))
    U, V = singular_value_decomposition(X_replace_nans, d)
    valid_prediction = predict(U, V, d, valid_pairs, False)
    train_prediction = predict(U, V, d, training_pairs, True)
    mse = mean_squared_error(train_prediction, training_labels)
    print("MSE = {0}".format(mse))
    accuracy_score = metrics.accuracy_score(valid_labels, valid_prediction)
    print("Accuracy score = {0}".format(accuracy_score))

# MSE
for lam in [0.1, 1, 10, 100, 400]:
    for d in [2, 5, 10, 20]:
        print("d = {0}, lambda = {1}".format(d, lam))
        U, V = matrix_completion(X, d, lam)
        valid_prediction = predict(U, V, d, valid_pairs, False)
        train_prediction = predict(U, V, d, training_pairs, True)
        mse = mean_squared_error(train_prediction, training_labels)
        print("MSE = {0}".format(mse))
        accuracy_score = metrics.accuracy_score(valid_labels, valid_prediction)
        print("Accuracy score = {0}".format(accuracy_score))

# Kaggle submission
lam = 400
d = 10
U, V = matrix_completion(X, d, lam)
test_prediction = predict(U, V, d, test_pairs, False)

outfile = open('./kaggle_submission.txt', 'w')
writer = csv.writer(outfile)
writer.writerow(['Id', 'Category'])
for i in range(len(test_prediction)):
    writer.writerow([int(i+1), int(test_prediction[i])])
outfile.close()
