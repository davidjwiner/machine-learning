#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:31:49 2016

@author: davidjwiner
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def train_clusters(X_train, num_clusters):
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    X_clusters = np.random.randint(0, num_clusters, size=n)
    next_X_clusters = np.zeros(n)
    means = np.zeros((d, num_clusters))
    
    iter_num = 0
        
    while not np.allclose(X_clusters, next_X_clusters):
        iter_num += 1
        print("Num clusters = {0}, iteration = {1}".format(num_clusters, iter_num))
        if (iter_num > 1):
            X_clusters = next_X_clusters
        for i in range(num_clusters):
            curr_cluster = X_train[X_clusters == i]
            means[:, i] = np.sum(curr_cluster, axis=0)/curr_cluster.shape[0]
        next_X_clusters = np.apply_along_axis(find_best_cluster, 1, X_train, means)
        
        if (iter_num % 25 == 0):
            np.save('centers_{0}.npy'.format(k), means)
    
    return means
    
# Utility function that calculates optimal cluster for a given example 
def find_best_cluster(x, cluster_means):
    k = cluster_means.shape[1]
    x_broadcast = np.tile(np.matrix(x), (k, 1))
    sum_squared_dist = np.linalg.norm(cluster_means - x_broadcast.T, axis=0)
    return np.argmin(sum_squared_dist)

def plot(centers, k):
    centers = centers * 255.0
    cols = 4
    rows = np.int(np.ceil(k / 4.0))
    for i in range(k):
        col = i % 4
        row = np.int(np.floor(i / 4.0))
        ax = plt.subplot2grid((rows, cols),(row, col))
        ax.imshow(centers[:,:,i], cmap='gray')
        ax.axis('off')
    plt.show()
    plt.savefig('centers_plot_{0}'.format(k))
    
if __name__ == "__main__":
    images = sio.loadmat('./mnist_data/images.mat')['images']
    X_train = (images.reshape((28*28, 60000))).T
    k_vals = [5, 10, 20]
    for k in k_vals:
        centers = train_clusters(X_train, k)
        # save to disk
        np.save('centers_{0}.npy'.format(k), centers)
        centers = np.load('centers_{0}.npy'.format(k)).reshape((28, 28, k))
        plot(centers, k)    
    
    
    
