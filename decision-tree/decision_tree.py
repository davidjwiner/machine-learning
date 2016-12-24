#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 23:29:23 2016

@author: davidjwiner
"""

import scipy.io as sio
import scipy.stats as stats
import numpy as np
import sklearn.metrics as metrics
import csv
from census_featurize import load_census_data
import pickle
from sklearn.cross_validation import train_test_split

def load_spam_dataset():
    return sio.loadmat('./spam_data/spam_data.mat')

class DecisionTree:
    
    def __init__(self, depth):
        self.depth = depth
        self.root = None
        return
    
    def impurity(self, left_label_hist, right_label_hist):
        left_entropy = self.entropy(left_label_hist)
        right_entropy = self.entropy(right_label_hist)
        
        n_left = np.sum(left_label_hist[:, 1])
        n_right = np.sum(right_label_hist[:, 1])
        n_tot = n_left + n_right
        
        return ((n_left * left_entropy) + (n_right * right_entropy)) / n_tot
    
    def entropy(self, hist):
        total_examples = np.sum(hist[:,1])
        probabilities = hist[:,1] / total_examples
        log_probabilities = np.log(probabilities)
        return -1 * np.sum(probabilities * log_probabilities)

    def segmenter(self, data, labels):
        dims = data.shape[1]
        min_impurity = 1e9
        dec_rule = (None, None)
        
        labels = np.squeeze(labels)

        for d in range(dims):
            dim = np.squeeze(data[:, d])
            temp = np.array(dim)
            unique_vals = np.unique(temp)
            unique_vals_matrix = np.matrix(unique_vals)
            impurities = np.apply_along_axis(self.get_impurity, 0, unique_vals_matrix, labels, dim)
            best_impurity_idx = np.argmin(impurities)
            best_impurity = impurities[best_impurity_idx]
            best_val = unique_vals[best_impurity_idx]
            if (best_impurity < min_impurity):
                min_impurity = best_impurity
                dec_rule = (d, best_val)
    
        d = dec_rule[0]
        v = dec_rule[1]
        dim = np.squeeze(data[:, d])
                
        l_indices = dim > v
        r_indices = dim <= v
        
        lX = data[l_indices, :]
        ly = labels[l_indices]
        
        rX = data[r_indices, :]
        ry = labels[r_indices]
        
        return lX, ly, rX, ry, dec_rule
    
    def get_impurity(self, val, labels, dim):
        greater = labels[dim > val]
        less_eq = labels[dim <= val]
        left_hist = stats.itemfreq(greater)
        right_hist = stats.itemfreq(less_eq)
        return self.impurity(left_hist, right_hist)

    def train(self, X, y, curr_depth):
        hist = stats.itemfreq(y)
        entropy = self.entropy(hist)
        if (curr_depth == self.depth or entropy < 0.00001 or y.size == 1):
            try:
                mode = stats.mode(y)[0][0]
                return Node(None, None, None, mode)
            except IndexError:
                return None
        else:
            lX, ly, rX, ry, dec_rule = self.segmenter(X, y)
            next_depth = curr_depth + 1
            node = Node(dec_rule, self.train(lX, ly, next_depth), self.train(rX, ry, next_depth), None)
            if (curr_depth == 0):
                self.root = node
            return node
                        
    def predict_one_ex(self, x):
        currNode = self.root
        while(True):
            if (currNode.label != None):
                return currNode.label
            else:
                d = currNode.split_rule[0]
                v = currNode.split_rule[1]
                if (x[d] > v):
                    currNode = currNode.left
                else:
                    currNode = currNode.right
    
    def predict(self, X):
        y = np.apply_along_axis(self.predict_one_ex, 1, X)
        return y
    
    def predict_one_ex_verbose(self, x, outfile):
        currNode = self.root
        writer = csv.writer(outfile)
        writer.writerow(['Dimension', 'Operator', 'Split value'])
        while(True):
            if (currNode.label != None):
                return currNode.label
                writer.writerow(['End, class: ', currNode.label])
            else:
                d = currNode.split_rule[0]
                v = currNode.split_rule[1]
                if (x[d] > v):
                    writer.writerow([d, '>', v])
                    currNode = currNode.left
                else:
                    currNode = currNode.right
                    writer.writerow([d, '<=', v])

class Node:
    
    def __init__(self, split_rule, left, right, label):
        self.split_rule = split_rule
        self.left = left
        self.right = right
        self.label = label

def random_forest(num_trees, depth, X, y, verbose=False):
    
    n_train = X.shape[0]
    forest = []
        
    for j in range(num_trees):
        print("##### Tree number {0} ######".format(j))
        sample_idxs = np.random.binomial(1, 0.5, n_train)
        tree = DecisionTree(depth)
        data = X[sample_idxs > 0, :]
        labels = y[sample_idxs > 0]
        tree.train(data, labels, 0)
        forest.append(tree)
        if(verbose):
            rule = tree.root.split_rule
            print("Root decision rule: Dimension {0}, Value {1}".format(rule[0], rule[1]))
    
    return forest

def random_forest_predict(forest, X):
    n = X.shape[0]
    predictions = np.zeros((n, 2))
    
    for tree in forest:
        try:
            pred_labels = tree.predict(X)
        except AttributeError:
            continue
        predictions[pred_labels == 0, 0] += 1
        predictions[pred_labels == 1, 1] += 1
    
    return np.argmax(predictions, axis=1)

def train_and_write_census(kaggle=False):
    training, testing, training_labels = load_census_data()
    X_train = np.asarray(training.todense())
    X_test = np.asarray(testing.todense())
    y_train = np.asarray(training_labels)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, np.squeeze(y_train), test_size = 0.2, random_state=0)
    
    tree = DecisionTree(20)
        
    tree.segmenter(X_train, y_train)
    tree.train(X_train, y_train, 0)
    pred_train_labels = tree.predict(X_train)
    true_train_labels = np.squeeze(y_train)
    pred_val_labels = tree.predict(X_val)
    true_val_labels = np.squeeze(y_val)
    
    accuracy_score = metrics.accuracy_score(true_train_labels, pred_train_labels)
    accuracy_score_val = metrics.accuracy_score(true_val_labels, pred_val_labels)
    print("Single tree training accuracy: {0}\n".format(accuracy_score))
    print("Single tree validation accuracy: {0}\n".format(accuracy_score_val))
    
#     Save for later since tree takes a little while to train
    with open('census_tree_1.pkl', 'wb') as output:
        pickle.dump(tree, output, pickle.HIGHEST_PROTOCOL)

    num_trees = 5
    depth = 15

    forest = random_forest(num_trees, depth, X_train, y_train, True)
    
    with open('census_forest_2.pkl', 'wb') as output:
        pickle.dump(forest, output, pickle.HIGHEST_PROTOCOL)
    
    pred_train_labels = random_forest_predict(forest, X_train)
    true_train_labels = np.squeeze(y_train)
    pred_val_labels = random_forest_predict(forest, X_val)
    true_val_labels = np.squeeze(y_val)
    accuracy_score = metrics.accuracy_score(true_train_labels, pred_train_labels)
    accuracy_score_val = metrics.accuracy_score(true_val_labels, pred_val_labels)
    print("Ensembled training accuracy: {0}\n".format(accuracy_score))
    print("Ensembled validation accuracy: {0}\n".format(accuracy_score_val))
    
     # Save for later since tree takes a little while to train
     
    file = open("census_forest_2.pkl",'rb')
    census_forest = pickle.load(file)
    file.close()
    
    for tree in census_forest:
        curr_root = tree.root
        print(curr_root.split_rule)
         
    if kaggle:
        census_pred_labels_test = random_forest_predict(census_forest, X_test)
        outfile = open('./census-output-data-forest-2.csv', 'w')
        writer = csv.writer(outfile)
        writer.writerow(['Id', 'Category'])
        for i in range(len(census_pred_labels_test)):
            writer.writerow([int(i+1), int(census_pred_labels_test[i])])
        outfile.close()

def train_and_write_spam(kaggle=False):
    data = load_spam_dataset()
    X_test = data['test_data']
    X_train = data['training_data']
    y_train = data['training_labels']

    X_train, X_val, y_train, y_val = train_test_split(X_train, np.squeeze(y_train), test_size = 0.2, random_state=0)
    
    tree = DecisionTree(20)
        
    tree.segmenter(X_train, y_train)
    tree.train(X_train, y_train, 0)
    pred_train_labels = tree.predict(X_train)
    true_train_labels = np.squeeze(y_train)
    pred_val_labels = tree.predict(X_val)
    true_val_labels = np.squeeze(y_val)
    accuracy_score = metrics.accuracy_score(true_train_labels, pred_train_labels)
    accuracy_score_val = metrics.accuracy_score(true_val_labels, pred_val_labels)
    print("Single tree training accuracy: {0}\n".format(accuracy_score))
    print("Single tree validation accuracy: {0}\n".format(accuracy_score_val))
    
    num_trees = 10
    depth = 20
        
    forest = random_forest(num_trees, depth, X_train, y_train, False)
    pred_train_labels = random_forest_predict(forest, X_train)
    true_train_labels = np.squeeze(y_train)
    pred_val_labels = random_forest_predict(forest, X_val)
    true_val_labels = np.squeeze(y_val)
    accuracy_score = metrics.accuracy_score(true_train_labels, pred_train_labels)
    accuracy_score_val = metrics.accuracy_score(true_val_labels, pred_val_labels)
    print("Ensembled training accuracy: {0}\n".format(accuracy_score))
    print("Ensembled validation accuracy: {0}\n".format(accuracy_score_val))
    
    if kaggle:
        num_trees = 15
        depth = 40
        
        pred_labels_test = random_forest(num_trees, depth, X_train, y_train, X_test, False)
        
        outfile = open('./spam-output-data2.csv', 'w')
        writer = csv.writer(outfile)
        writer.writerow(['Id', 'Category'])
        for i in range(len(pred_labels_test)):
            writer.writerow([int(i+1), int(pred_labels_test[i])])
        outfile.close()

train_and_write_census(True)
train_and_write_spam(True)