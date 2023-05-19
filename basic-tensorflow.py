import tensorflow as tf
import numpy as np

# Generate random input data
X = np.random.rand(100).astype(np.float32)

# Generate corresponding output data with a linear relationship
Y = 2 * X + 1

# Initialize model parameters
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

# Define the linear regression model
def linear_regression(x):
    return W * x + b

# Define loss function (mean squared error)
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define the optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)


# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    # Perform forward pass
    with tf.GradientTape() as tape:
        y_pred = linear_regression(X)
        loss = mean_squared_error(Y, y_pred)
    
    # Calculate gradients
    gradients = tape.gradient(loss, [W, b])
    
    # Update model parameters
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.numpy():.4f}")

# Generate test data
X_test = np.array([0.2, 0.5, 0.8], np.float32)

# Predict the outputs for the test data
Y_test = linear_regression(X_test)

# Print the predictions
print("Test Predictions:")
for i in range(len(X_test)):
    print(f"Input: {X_test[i]:.2f}, Predicted Output: {Y_test[i]:.2f}")
