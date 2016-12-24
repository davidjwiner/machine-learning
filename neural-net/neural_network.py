from mnist import MNIST
import numpy as np
import sklearn.metrics as metrics
import sklearn
import matplotlib.pyplot as plt
import csv

N_HID = 200
N_OUT = 10

def train_neural_network(X, labels, step_size, decay, num_iter, outfile):
    
    mu, sigma = 0, 0.01
    n_dim = X.shape[1]
    n_ex = X.shape[0]
    
    W = np.random.normal(mu, sigma, (N_OUT, N_HID + 1))
    V = np.random.normal(mu, sigma, (N_HID, n_dim + 1))
    
    step = step_size
    
    iterations = []
    losses = []
    accuracies = []
    
    for i in range(num_iter):
        if (i % 100000 == 0):
            step = step * decay
        ex_num = i % n_ex
        Y = np.matrix(labels[ex_num, :])
        
        # Pull X_0 layer and append one for bias term
        X_0 = np.matrix(np.insert(X[ex_num, :], 0, 1))
        
        # Forward propagate from input to hidden layer, again adding bias term
        Z_0 = np.dot(V, X_0.T)
        X_1 = np.maximum(Z_0, 0) # ReLU
        X_1 = np.insert(X_1.T, 0, 1)
        
        # Forward propagate from hidden layer to output layer
        Z_1 = np.dot(W, X_1.T).T
        X_2 = np.apply_along_axis(softmax, 1, Z_1, np.arange(10))
        
        # W grad calculations
        y_sums = np.tile(1, 10) - Y
        deltas = np.multiply(X_2, y_sums) - np.multiply(Y, (1 - X_2))
        grad_W = np.dot(deltas.T, X_1)        
        
        # V grad calculations
        W_trim = W[:, 1:] # don't pass bias term along to next layer down
        gammas = np.dot(W_trim.T, deltas.T)
        multiplier = np.maximum(Z_0, 0)/Z_0
        gammas = np.multiply(multiplier, gammas)
        grad_V = np.dot(gammas, X_0)
                
        # W and V updates
        W = W - step * grad_W
        V = V - step * grad_V
        
        
        if (i % 5000 == 0):
            pred_labels = predict_neural_network(X, V, W)
            true_labels = np.argmax(labels, axis = 1)
            accuracy_score = metrics.accuracy_score(true_labels, pred_labels)
            print("Train accuracy: {0}\n".format(accuracy_score))
            outfile.write("Train accuracy: {0}\n".format(accuracy_score))
            training_loss = 0
            for k in range(n_ex):
                myX = np.matrix(np.insert(X[k, :], 0, 1))
                myY = np.matrix(labels[k, :])
                training_loss += x_entropy_loss(myX, myY, W, V)
            print("Training loss is {0}".format(training_loss))
            iterations = np.append(iterations, i)
            losses = np.append(losses, training_loss)
            accuracies = np.append(accuracies, accuracy_score)
            
    np.save("W_save.npy", W)
    np.save("V_save.npy", V)
    np.save("iterations.npy", iterations)
    np.save("losses.npy", losses)
    np.save("accuracies.npy", accuracies)
    return V, W

def x_entropy_loss(X, Y, W, V):
    # Input X already has bias term appended
    # Forward propagate from input layer to hidden layer
    Z_0 = np.dot(V, X.T)
    X_1 = np.maximum(Z_0, 0) # ReLU
    X_1 = np.insert(X_1.T, 0, 1)
    
    # Forward propagate from hidden layer to output layer
    Z_1 = np.dot(W, X_1.T).T
    X_2 = np.apply_along_axis(softmax, 1, np.matrix(Z_1), np.arange(10))
    
    log_output = np.log(X_2)
    return -1 * np.sum(np.multiply(Y, log_output))
        
def predict_neural_network(X, V, W):
    n_ex = X.shape[0]

    # Pull X_0 layer and append one for bias term
    X_0 = np.insert(X, 0, np.ones(n_ex), axis=1)
    Z_0 = np.dot(V, X_0.T)
    
    # Forward propagate from input to hidden layer, again adding bias term
    X_1 = np.maximum(Z_0, 0) # ReLU
    X_1 = np.insert(X_1, 0, np.ones(n_ex), axis=0)
    Z_1 = np.dot(W, X_1)
    
    # Forward propagate from hidden layer to output layer
    X_2 = np.apply_along_axis(softmax, 0, Z_1, np.arange(10))
    
    # Return highest prediction
    return np.argmax(X_2, axis = 0)

def softmax(z, j):
    # Uses numerical stability trick
    b = np.max(z)
    denom = np.sum(np.exp(z - b))
    num = np.exp(z[j] - b)
    return num/denom

def load_dataset():
    mndata = MNIST('./data/')
    X, labels = map(np.array, mndata.load_training())
    
    # Shuffle data
    X_shuf, labels_shuf = sklearn.utils.shuffle(X, labels, random_state = 40) 
    
    # Split data into training and validation sets
    X_train = X_shuf[0:50000, :]
    labels_train = one_hot(labels_shuf[0:50000])
    X_val = X_shuf[50000:, :]
    labels_val = one_hot(labels_shuf[50000:])

    # The test labels are meaningless,
    # since you're replacing the official MNIST test set with our own test set
    X_test, _ = map(np.array, mndata.load_testing())
    
    # Center and normalize data
    X_train = standardize(X_train)
    X_val = standardize(X_val)
    X_test = standardize(X_test)
    
    # Save for later use, which is faster than doing all this preprocessing every time
    np.save("X_test.npy", X_test)
    np.save("X_train.npy", X_train)
    np.save("labels_train.npy", labels_train)
    
    return X_train, labels_train, X_val, labels_val, X_test

def standardize(X):
    global_mean = np.sum(X)/X.size
    return (X - global_mean)/255

def one_hot(labels_train):
    z = np.zeros((labels_train.shape[0], N_OUT))
    for i in range(len(labels_train)):
        digit = labels_train[i]
        z[i, digit] = 1
    return z
    
X_train, labels_train, X_val, labels_val, X_test = load_dataset()

# These commented out lines were used when I didn't want to reload every time
# I ran training
# X_train = np.load("X_train.npy")
# labels_train = np.load("labels_train.npy")

# Tested other alpha and decay values in a grid (see write-up); these are the ones I landed on
alphas = [0.01]
decays = [0.9]

outfile = open("ResultsFile9.txt", "w")

for alpha in alphas:
    for decay in decays:
        print("##### alpha = {0}, decay = {1} ######".format(alpha, decay))
        outfile.write("##### alpha = {0}, decay = {1} ######\n".format(alpha, decay))
        V, W = train_neural_network(X_train, labels_train, alpha, decay, 200000, outfile)

outfile.close()
predict_neural_network(X_train, V, W)

# Plotting
iterations = np.load("iterations.npy")
losses = np.load("losses.npy")
accuracies = np.load("accuracies.npy")

fig = plt.figure(0)
plt.plot(iterations, accuracies)
plt.xlabel("SGD iteration")
plt.ylabel("Training classification accuracy")

W = np.load("W_save.npy")
V = np.load("V_save.npy")

# Prediction on validation set
pred_labels_val = predict_neural_network(X_val, V, W)
true_labels_val = np.argmax(labels_val, axis = 1)
accuracy_score = metrics.accuracy_score(true_labels_val, pred_labels_val)
print("Validation accuracy: {0}\n".format(accuracy_score))

# Prediction on test set
pred_labels_test = predict_neural_network(X_test, V, W)

outfile = open('./output-data.csv', 'w')
writer = csv.writer(outfile)
writer.writerow(['Id', 'Category'])
for i in range(len(pred_labels_test)):
    writer.writerow([int(i+1), int(pred_labels_test[i])])
outfile.close()


