#%%



#best settings found are 0.2 learning rate and batch size = 128
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

# Reshape data to 1D vector
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

# Normalize pixel values between 0 and 1
train_X = train_X / 255.0
test_X = test_X / 255.0

# Convert labels to one-hot vectors
num_classes = 10
train_y = np.eye(num_classes)[train_y]
test_y = np.eye(num_classes)[test_y]

# Set hyperparameters
learning_rate = 0.2
epochs = 10
batch_size = 128

# Initialize weights and biases
input_size = train_X.shape[1]
output_size = num_classes
hidden_size = 256

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Define ReLU activation function
def relu(z):
    return np.maximum(0, z)

# Define softmax activation function
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Define cross-entropy loss function
def cross_entropy_loss(y_hat, y):
    m = y.shape[0]
    loss = -np.sum(y * np.log(y_hat)) / m
    return loss

# Define accuracy function
def accuracy(y_hat, y):
    pred = np.argmax(y_hat, axis=1)
    true = np.argmax(y, axis=1)
    acc = np.mean(pred == true)
    return acc

# Train model
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    # Shuffle data
    permutation = np.random.permutation(train_X.shape[0])
    train_X = train_X[permutation]
    train_y = train_y[permutation]

    # Mini-batch training
    for i in range(0, train_X.shape[0], batch_size):
        # Forward pass
        X_batch = train_X[i:i+batch_size]
        y_batch = train_y[i:i+batch_size]
        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y_hat = softmax(z2)

        # Backward pass
        dL = y_hat - y_batch
        dW2 = np.dot(a1.T, dL) / batch_size
        db2 = np.sum(dL, axis=0, keepdims=True) / batch_size
        dL1 = np.dot(dL, W2.T) * (z1 > 0)
        dW1 = np.dot(X_batch.T, dL1) / batch_size
        db1 = np.sum(dL1, axis=0, keepdims=True) / batch_size
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Evaluate model on training and test sets
    z_train1 = np.dot(train_X, W1) + b1
    a_train1 = relu(z_train1)
    z_train2 = np.dot(a_train1, W2) + b2
    y_hat_train = softmax(z_train2)
    train_loss.append(cross_entropy_loss(y_hat_train, train_y))
    train_acc.append(accuracy(y_hat_train, train_y))

    z_test1 = np.dot(test_X, W1) + b1
    a_test1 = relu(z_test1)
    z_test2 = np.dot(a_test1, W2) + b2
    y_hat_test = softmax(z_test2)
    test_loss.append(cross_entropy_loss(y_hat_test, test_y))
    test_acc.append(accuracy(y_hat_test, test_y))

    # Print loss and accuracy
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f} - Train Acc: {train_acc[-1]:.4f} - Test Loss: {test_loss[-1]:.4f} - Test Acc: {test_acc[-1]:.4f}")

# Plot the loss and accuracy curves over epochs
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(range(epochs), train_loss, 'r', label='Training Loss')
axs[0].plot(range(epochs), test_loss, 'b', label='Test Loss')
axs[0].set_title('Training and Test Loss')
axs[0].legend(loc='upper right')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')

axs[1].plot(range(epochs), train_acc, 'r', label='Training Accuracy')
axs[1].plot(range(epochs), test_acc, 'b', label='Test Accuracy')
axs[1].set_title('Training and Test Accuracy')
axs[1].legend(loc='lower right')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')

plt.show()




#%%