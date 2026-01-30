import sys

# Check for TensorFlow/Keras availability
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("Error: TensorFlow is not installed.")
    print("Please install it with: pip install tensorflow")
    sys.exit(1)

import numpy as np

# Load MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape data to add channel dimension (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")

# Build the model
# Architecture: Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Dense -> Dropout -> Dense
print("\nBuilding model...")

model = keras.Sequential([
    # First convolutional layer - 32 filters, 3x3 kernel
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Second convolutional layer - 64 filters, 3x3 kernel
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten the 3D output to 1D
    layers.Flatten(),
    
    # Hidden layer with 128 units
    layers.Dense(128, activation="relu"),
    
    # Dropout layer with 50% dropout rate
    layers.Dropout(0.5),
    
    # Output layer with 10 units (one for each digit 0-9)
    layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Train the model
print("\nTraining model for 10 epochs...")
history = model.fit(
    x_train, 
    y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Save the model
model_filename = "handwriting_model.h5"
print(f"\nSaving model to {model_filename}...")
model.save(model_filename)
print("Model saved successfully!")

print("\nTraining complete!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")