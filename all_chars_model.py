import sys
import os
import numpy as np
import json

# Check for TensorFlow/Keras availability
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_datasets as tfds
except ImportError:
    print("Error: TensorFlow is not installed.")
    print("Please install it with: pip install tensorflow tensorflow-datasets")
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


def get_available_datasets():
    """Return list of available TensorFlow image classification datasets"""
    return {
        '1': {'name': 'MNIST', 'key': 'mnist', 'classes': 10, 'desc': 'Handwritten digits (0-9)'},
        '2': {'name': 'Fashion-MNIST', 'key': 'fashion_mnist', 'classes': 10, 'desc': 'Fashion items'},
        '3': {'name': 'EMNIST Letters', 'key': 'emnist/letters', 'classes': 27, 'desc': 'Handwritten letters (A-Z + blank)'},
        '4': {'name': 'EMNIST Balanced', 'key': 'emnist/balanced', 'classes': 47, 'desc': 'Digits + letters (balanced)'},
        '5': {'name': 'EMNIST Digits', 'key': 'emnist/digits', 'classes': 10, 'desc': 'Handwritten digits'},
        '6': {'name': 'KMNIST', 'key': 'kmnist', 'classes': 10, 'desc': 'Japanese characters'},
        '7': {'name': 'Custom Dataset', 'key': 'custom', 'classes': None, 'desc': 'Your own character images'}
    }


def load_tensorflow_dataset(dataset_key):
    """Load a dataset from TensorFlow Datasets"""
    print(f"\nLoading {dataset_key} dataset...")
    
    try:
        if dataset_key in ['mnist', 'fashion_mnist', 'kmnist']:
            # Use keras.datasets for these
            if dataset_key == 'mnist':
                (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            elif dataset_key == 'fashion_mnist':
                (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
            else:  # kmnist
                # Load using tfds
                ds_train, ds_test = tfds.load(dataset_key, split=['train', 'test'], as_supervised=True)
                
                # Convert to numpy
                x_train = np.array([x.numpy() for x, _ in ds_train])
                y_train = np.array([y.numpy() for _, y in ds_train])
                x_test = np.array([x.numpy() for x, _ in ds_test])
                y_test = np.array([y.numpy() for _, y in ds_test])
        else:
            # Use tfds for EMNIST and others
            ds_train, ds_test = tfds.load(dataset_key, split=['train', 'test'], as_supervised=True)
            
            # Convert to numpy
            x_train = np.array([x.numpy() for x, _ in ds_train])
            y_train = np.array([y.numpy() for _, y in ds_train])
            x_test = np.array([x.numpy() for x, _ in ds_test])
            y_test = np.array([y.numpy() for _, y in ds_test])
        
        # Normalize
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        # Reshape to add channel dimension
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
        
        print(f"Training samples: {x_train.shape[0]}")
        print(f"Test samples: {x_test.shape[0]}")
        print(f"Image shape: {x_train.shape[1:]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        return x_train, y_train, x_test, y_test, len(np.unique(y_train))
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure tensorflow-datasets is installed: pip install tensorflow-datasets")
        return None, None, None, None, None


def load_user_images(data_dir="character_data"):
    if not PIL_AVAILABLE:
        return None, None, None, None
    
    if not os.path.exists(data_dir):
        return None, None, None, None
    
    images = []
    labels = []
    class_names = []
    
    # Supported image formats
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    
    print(f"\nLoading user images from '{data_dir}'...")
    
    # Get all subdirectories (class folders)
    class_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    
    if not class_dirs:
        return None, None, None, None
    
    print(f"Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
    
    # Iterate through each class folder
    for class_idx, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_dir, class_name)
        class_names.append(class_name)
        
        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_dir) 
                      if os.path.splitext(f.lower())[1] in valid_extensions]
        
        if not image_files:
            continue
        
        print(f"  Loading {len(image_files)} images for class '{class_name}'...")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_dir, img_file)
                
                # Load and process image
                img = Image.open(img_path)
                
                # Convert to grayscale
                img_gray = img.convert('L')
                
                # Resize to 28x28
                img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                img_array = np.array(img_resized, dtype=np.float32)
                
                # Invert if needed (white background with black character -> black background with white character)
                if np.mean(img_array) > 127:
                    img_array = 255 - img_array
                
                # Normalize to [0, 1]
                img_array = img_array / 255.0
                
                # Add to lists
                images.append(img_array)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"    Warning: Could not load {img_file}: {e}")
                continue
    
    if not images:
        return None, None, None, None
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Add channel dimension
    images = np.expand_dims(images, -1)
    
    num_classes = len(class_names)
    
    print(f"  Successfully loaded {len(images)} user images")
    print(f"  Classes: {class_names}")
    
    return images, labels, class_names, num_classes


def combine_datasets(data1_x, data1_y, data2_x, data2_y):
    """Combine two datasets"""
    if data2_x is None or len(data2_x) == 0:
        return data1_x, data1_y
    
    combined_x = np.concatenate([data1_x, data2_x], axis=0)
    combined_y = np.concatenate([data1_y, data2_y], axis=0)
    
    # Shuffle the combined dataset
    indices = np.random.permutation(len(combined_x))
    combined_x = combined_x[indices]
    combined_y = combined_y[indices]
    
    return combined_x, combined_y


def save_class_mapping(class_names, filename="class_mapping.json"):
    """Save class names to a JSON file"""
    mapping = {str(i): name for i, name in enumerate(class_names)}
    with open(filename, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Class mapping saved to {filename}")


# Main training script
print("="*70)
print("Universal Character Recognition - Training")
print("="*70)

# Display available datasets
datasets = get_available_datasets()
print("\nAvailable datasets:")
for key, info in datasets.items():
    print(f"{key}. {info['name']}: {info['desc']}")

# Ask user to select dataset
while True:
    try:
        dataset_choice = input("\nSelect dataset (1-7) [default: 1]: ").strip()
        if dataset_choice == "":
            dataset_choice = "1"
        if dataset_choice in datasets:
            break
        print("Invalid choice. Please enter 1-7.")
    except KeyboardInterrupt:
        print("\n\nTraining cancelled.")
        sys.exit(0)

selected_dataset = datasets[dataset_choice]

# Initialize variables
x_train = None
y_train = None
x_test = None
y_test = None
num_classes = None
class_names = None
user_x_train = None
user_y_train = None
use_augmentation = False

# Load selected dataset
if selected_dataset['key'] != 'custom':
    # Load TensorFlow dataset
    x_train, y_train, x_test, y_test, num_classes = load_tensorflow_dataset(selected_dataset['key'])
    
    if x_train is None:
        print("Failed to load dataset. Exiting.")
        sys.exit(1)
    
    # Ask if user wants to add custom images
    print("\nDo you want to add custom images to augment this dataset?")
    augment_choice = input("Add custom images? (y/n) [default: n]: ").strip().lower()
    
    if augment_choice in ['y', 'yes']:
        use_augmentation = True
        user_x_train, user_y_train, user_class_names, user_num_classes = load_user_images("character_data")
        
        if user_x_train is not None:
            # Check if classes match
            if user_num_classes != num_classes:
                print(f"\nWarning: User dataset has {user_num_classes} classes, but {selected_dataset['name']} has {num_classes} classes.")
                print("Classes must match for augmentation. Skipping user images.")
                user_x_train = None
            else:
                print(f"\nWill augment {selected_dataset['name']} with {len(user_x_train)} custom images")
else:
    # Load only custom dataset
    user_x_train, user_y_train, class_names, num_classes = load_user_images("character_data")
    
    if user_x_train is None:
        print("\nError: No custom images found in 'character_data' folder.")
        print("Please organize your images in class subfolders.")
        print("\nExample structure:")
        print("  character_data/")
        print("    A/image1.png")
        print("    B/image1.png")
        print("    0/image1.png")
        print("    1/image1.png")
        print("    cat/image1.png")
        print("    ...")
        sys.exit(1)
    
    # Split custom data into train/test (80/20)
    split_idx = int(0.8 * len(user_x_train))
    indices = np.random.permutation(len(user_x_train))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    x_train = user_x_train[train_indices]
    y_train = user_y_train[train_indices]
    x_test = user_x_train[test_indices]
    y_test = user_y_train[test_indices]
    
    print(f"\nCustom dataset split:")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")

# Combine datasets if augmenting
if use_augmentation and user_x_train is not None:
    original_train_size = len(x_train)
    x_train, y_train = combine_datasets(x_train, y_train, user_x_train, user_y_train)
    print(f"\nCombined training samples: {len(x_train)}")
    print(f"  - Original dataset: {original_train_size}")
    print(f"  - Custom images: {len(user_x_train)}")

print(f"\nFinal training set shape: {x_train.shape}")
print(f"Final test set shape: {x_test.shape}")
print(f"Number of classes: {num_classes}")

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
    
    # Output layer with dynamic number of units based on classes
    layers.Dense(num_classes, activation="softmax")
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

# Generate model filename
if selected_dataset['key'] == 'custom':
    model_filename = "character_model_custom.h5"
    mapping_filename = "class_mapping_custom.json"
elif use_augmentation:
    dataset_name = selected_dataset['key'].replace('/', '_')
    model_filename = f"character_model_{dataset_name}_augmented.h5"
    mapping_filename = f"class_mapping_{dataset_name}.json"
else:
    dataset_name = selected_dataset['key'].replace('/', '_')
    model_filename = f"character_model_{dataset_name}.h5"
    mapping_filename = f"class_mapping_{dataset_name}.json"

# Save the model
print(f"\nSaving model to {model_filename}...")
model.save(model_filename)
print("Model saved successfully!")

# Save class mapping if we have class names
if class_names is not None:
    save_class_mapping(class_names, mapping_filename)

print("\n" + "="*70)
print("Training complete!")
print("="*70)
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Model saved as: {model_filename}")

if class_names:
    print(f"Class mapping saved as: {mapping_filename}")
    print(f"Classes: {', '.join(class_names)}")

print("\nYou can now use this model with the recognition interface!")
print("Make sure to update the model filename in recognize.py or recognize_with_images.py")