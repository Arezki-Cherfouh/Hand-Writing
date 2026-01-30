import sys
import os
import numpy as np

# Check for TensorFlow/Keras availability
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("Error: TensorFlow is not installed.")
    print("Please install it with: pip install tensorflow")
    sys.exit(1)

# Check for PIL availability
try:
    from PIL import Image
except ImportError:
    print("Warning: PIL (Pillow) is not installed.")
    print("User image loading will not be available.")
    print("Install with: pip install pillow")
    PIL_AVAILABLE = False
else:
    PIL_AVAILABLE = True


def load_user_images(data_dir="training_data"):
    """
    Load user-provided images from organized folders
    
    Expected structure:
    training_data/
        0/
            image1.png
            image2.jpg
        1/
            image1.png
        ...
        9/
            image1.png
    
    Returns:
        images: numpy array of shape (n_samples, 28, 28, 1)
        labels: numpy array of shape (n_samples,)
    """
    if not PIL_AVAILABLE:
        return None, None
    
    if not os.path.exists(data_dir):
        return None, None
    
    images = []
    labels = []
    
    # Supported image formats
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    
    print(f"\nLoading user images from '{data_dir}'...")
    
    # Iterate through each digit folder (0-9)
    for digit in range(10):
        digit_dir = os.path.join(data_dir, str(digit))
        
        if not os.path.exists(digit_dir):
            continue
        
        # Get all image files in the digit folder
        image_files = [f for f in os.listdir(digit_dir) 
                      if os.path.splitext(f.lower())[1] in valid_extensions]
        
        if not image_files:
            continue
        
        print(f"  Loading {len(image_files)} images for digit {digit}...")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(digit_dir, img_file)
                
                # Load and process image
                img = Image.open(img_path)
                
                # Convert to grayscale
                img_gray = img.convert('L')
                
                # Resize to 28x28
                img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img_resized, dtype=np.float32)
                
                # Invert if needed (white background with black digit -> black background with white digit)
                if np.mean(img_array) > 127:
                    img_array = 255 - img_array
                
                # Normalize to [0, 1]
                img_array = img_array / 255.0
                
                # Add to lists
                images.append(img_array)
                labels.append(digit)
                
            except Exception as e:
                print(f"    Warning: Could not load {img_file}: {e}")
                continue
    
    if not images:
        return None, None
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Add channel dimension
    images = np.expand_dims(images, -1)
    
    print(f"  Successfully loaded {len(images)} user images")
    
    return images, labels


def combine_datasets(mnist_x, mnist_y, user_x, user_y):
    """Combine MNIST and user datasets"""
    if user_x is None or len(user_x) == 0:
        return mnist_x, mnist_y
    
    combined_x = np.concatenate([mnist_x, user_x], axis=0)
    combined_y = np.concatenate([mnist_y, user_y], axis=0)
    
    # Shuffle the combined dataset
    indices = np.random.permutation(len(combined_x))
    combined_x = combined_x[indices]
    combined_y = combined_y[indices]
    
    return combined_x, combined_y


# Main training script
print("="*60)
print("MNIST Handwritten Digit Recognition - Training")
print("="*60)

# Ask user about training mode
print("\nTraining modes:")
print("1. MNIST only (60,000 images)")
print("2. MNIST + User images (augmented)")
print("3. User images only (custom dataset)")

while True:
    try:
        mode = input("\nSelect mode (1/2/3) [default: 1]: ").strip()
        if mode == "":
            mode = "1"
        if mode in ["1", "2", "3"]:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    except KeyboardInterrupt:
        print("\n\nTraining cancelled.")
        sys.exit(0)

mode = int(mode)

# Load MNIST dataset
if mode in [1, 2]:
    print("\nLoading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Reshape data to add channel dimension (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"MNIST training samples: {x_train.shape[0]}")
    print(f"MNIST test samples: {x_test.shape[0]}")

# Load user images
user_x_train = None
user_y_train = None

if mode in [2, 3]:
    user_x_train, user_y_train = load_user_images("training_data")
    
    if user_x_train is None:
        if mode == 3:
            print("\nError: No user images found in 'training_data' folder.")
            print("Please organize your images in subfolders 0-9.")
            print("\nExample structure:")
            print("  training_data/")
            print("    0/image1.png")
            print("    1/image1.png")
            print("    ...")
            sys.exit(1)
        else:
            print("\nWarning: No user images found. Training with MNIST only.")
            mode = 1

# Prepare final training dataset
if mode == 1:
    # MNIST only
    final_x_train = x_train
    final_y_train = y_train
    final_x_test = x_test
    final_y_test = y_test
    
elif mode == 2:
    # MNIST + User images
    final_x_train, final_y_train = combine_datasets(x_train, y_train, 
                                                     user_x_train, user_y_train)
    final_x_test = x_test
    final_y_test = y_test
    print(f"\nCombined training samples: {final_x_train.shape[0]}")
    print(f"  - MNIST: {len(x_train)}")
    print(f"  - User images: {len(user_x_train)}")
    
elif mode == 3:
    # User images only
    final_x_train = user_x_train
    final_y_train = user_y_train
    
    # Split user data into train/test (80/20)
    split_idx = int(0.8 * len(user_x_train))
    indices = np.random.permutation(len(user_x_train))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    final_x_train = user_x_train[train_indices]
    final_y_train = user_y_train[train_indices]
    final_x_test = user_x_train[test_indices]
    final_y_test = user_y_train[test_indices]
    
    print(f"\nUser dataset split:")
    print(f"  Training samples: {len(final_x_train)}")
    print(f"  Test samples: {len(final_x_test)}")

print(f"\nFinal training set shape: {final_x_train.shape}")
print(f"Final test set shape: {final_x_test.shape}")

# Build the model
# Architecture: Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Dense -> Dropout -> Dense
print("\nBuilding model (CS50 AI Style)...")

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
    
    # Dropout layer with 50% dropout rate (CS50 AI style)
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
    final_x_train, 
    final_y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(final_x_test, final_y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Save the model
if mode == 1:
    model_filename = "handwriting_model.h5"
elif mode == 2:
    model_filename = "handwriting_model_augmented.h5"
else:
    model_filename = "handwriting_model_custom.h5"

print(f"\nSaving model to {model_filename}...")
model.save(model_filename)
print("Model saved successfully!")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Model saved as: {model_filename}")

if mode == 2:
    print(f"\nYour model has been trained on MNIST + {len(user_x_train)} custom images")
elif mode == 3:
    print(f"\nYour model has been trained on {len(final_x_train)} custom images only")
