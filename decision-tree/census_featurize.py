#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:25:53 2016

@author: davidjwiner
"""

import csv
import sklearn.feature_extraction as extraction
import pandas as pd


def load_data(path):
    df = pd.DataFrame.from_csv(path, header=0)
    cols = df.columns
    for col in cols:
        series = df[col]
        mode = series.mode()[0]
        try:
            df.ix[df[col]=='?', col] = mode
        except TypeError:
            continue
    return df

def load_census_data():
    df = load_data('./census_data/train_data.csv')
    labels = df.label
    y_train = labels.as_matrix()
    df = df.drop('label', 1)
    train_data = df.to_dict('records')
    vec = extraction.DictVectorizer()
    X_train = vec.fit_transform(train_data)
    outfile = open('./census_feature_names.csv', 'w')
    writer = csv.writer(outfile)
    i = 0
    features = vec.get_feature_names()
    for i in range(104):
        feat = features[i]
        writer.writerow([i, feat])
    df2 = load_data('./census_data/test_data.csv')
    test_data = df2.to_dict('records')
    X_test = vec.transform(test_data)
    
    return X_train, X_test, y_train

load_census_data()