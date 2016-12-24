Machine learning algorithms
===========================
This repository contains some simple implementations of classic machine learning algorithms I built for Berkeley's CS289A: Introduction to Machine Learning.

Neural net
----------
Here, I built a simple implementation of a three-layer fully connected neural network. This project helped me better understand the backpropagation algorithm, which I implemented in the enclosed code. The neural net is trained to recognize handwritten digits using the classic MNIST dataset. Even though my neural net was pretty simple, it still got the best prediction performance I've ever been able to achieve on this dataset (across all techniques), with ~98% validation accuracy.

Decision tree/random forest
---------------------------
In this model, I built a generalized decision tree that can be used to make predictions on all kinds of datasets. The two problems that I tested it on were:

* Spam prediction (~81% validation accuracy)
* Census data prediction (binary indicator of wealth based on other factors like marital status, occupation, education level, etc. — ~85% validation accuracy)

K-means
-------
Later in the class, we started to learn about some unsupervised learning algorithms. I liked k-means for its simplicity (and the fact that it works pretty well in practice even though the optimal solution is NP-hard). For this project, I used Lloyd's algorithm to cluster MNIST images. These are the results I got for 20 clusters:

![centers](/centers_plot_20.png?raw=true "20 clusters MNIST dataset")

Recommendation engine
---------------------
At the end of the class, we had a module on recommender systems and collaborative filtering. I used the singular value decomposition and matrix completion techniques to build a joke recommendation engine. The problem was pretty simple: given a couple thousand user ratings on 100 jokes, infer user archetypes (i.e., different "senses of humor") and map those archetypes to users and joke preferences. This all boils down to some pretty simple (yet powerful) matrix factorization.   