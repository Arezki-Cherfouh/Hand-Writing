This project is a handwriting recognition system using a Convolutional Neural Network trained on the MNIST dataset. The model is built with TensorFlow and Keras and recognizes digits from 0 to 9.

The training script loads and preprocesses MNIST, builds a CNN with convolution, pooling, dropout, and dense layers, then trains and saves the model.

The Pygame application provides a 32x32 drawing grid where users can draw digits with the mouse. The drawing is centered, resized to 28x28, and passed to the trained model for prediction. The predicted digit and confidence are displayed in real time.

Requirements include Python, TensorFlow, NumPy, Pygame, and SciPy. Train the model first, then run the GUI to start classifying handwritten digits.
