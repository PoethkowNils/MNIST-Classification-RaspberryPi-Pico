import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation

def create_lenet(input_shape=(28, 28, 1), num_classes=10):
    """
    Builds the LeNet architecture.
    
    Args:
        input_shape: tuple, shape of the input images (default: MNIST dimensions).
        num_classes: int, number of classes for the final classification layer.
        
    Returns:
        A Keras Sequential model representing LeNet.
    """
    model = Sequential()
    
    # First Convolutional Layer
    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), 
                     padding="same", input_shape=input_shape, name="conv1"))
    model.add(Activation("tanh"))  # Tanh activation (Maybe consider different one)
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1"))
    
    # Second Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), 
                     padding="valid", name="conv2"))
    model.add(Activation("tanh"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2"))
    
    # Flatten layer
    model.add(Flatten(name="flatten"))
    
    # First Fully Connected Layer
    model.add(Dense(units=120, activation="tanh", name="fc1"))
    
    # Second Fully Connected Layer
    model.add(Dense(units=84, activation="tanh", name="fc2"))
    
    # Output Layer
    model.add(Dense(units=num_classes, activation="softmax", name="output"))
    
    return model