# Tensorflow instances

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Sequential

# Shallow Model
# Build CNN model

def narrow_model(input_shape:tuple, num_classes:int)->Sequential:
    """
    Parameters
    ----------
    input_shape : tuple 
    num_classes : int 

    Returns
    ----------
    Sequential

    Notes
    ----------
    A basic CNN.
    """

    model = Sequential([

    Conv2D(64,3,padding='same',input_shape=(32,32,3), activation='relu'),
    BatchNormalization(),

    Conv2D(128,3,padding='same', activation='relu'),
    BatchNormalization(),

    Conv2D(128,3,padding='same', activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Dropout(0.2),



    Conv2D(256,3,padding='same', activation='relu'),
    BatchNormalization(),

    Conv2D(512,3,padding='same', activation='relu'),
    BatchNormalization(),

    Conv2D(512,3,padding='same', activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Dropout(0.25),


    Conv2D(768,3,padding='same', activation='relu'),
    BatchNormalization(),

    Conv2D(1024,3,padding='same', activation='relu'),
    BatchNormalization(),

    Conv2D(1024,3,padding='same', activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Dropout(0.25),

# Add a classifier on top of the CNN
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

    return model