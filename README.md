Machine learning algorithms
===========================

This repo contains some of the simple implementations of classic machine learning algorithms I built for Berkeley's CS289A: Introduction to Machine Learning. Quick overview of what I built:

Neural net
----------

Here, I built a simple implementation of a three-layer fully connected neural network. This project helped me better understand the backpropagation algorithm, which I implemented from scratch in the enclosed code. The neural net is trained to recognize handwritten digits using the classic MNIST dataset. Even though my neural net was pretty simple, it still got the best prediction performance I've ever been able to achieve on this dataset (across all techniques), with ~98% validation accuracy.

Decision tree/random forest
---------------------------

Here I built a generalized decision tree that can be used to make predictions on all kinds of datasets. The two problems that I tested it on were:

* Spam prediction (~81% validation accuracy)
* Census data prediction (binary indicator of wealth based on other factors like marital status, occupation, education level, etc. -- ~85% validation accuracy)

K-means
-------

Later in the class, we started to learn about some unsupervised learning algorithms. For this project, I took 
