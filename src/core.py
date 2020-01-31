from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import tensorflow as tf
mnist = tf.keras.datasets.mnist

def testModel():
    ''' test doc '''
    
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Define the Keras TensorBoard callback.
    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    model.fit(
        x_train, 
        y_train, 
        epochs=5,
        callbacks=[tensorboard_callback])
    model.evaluate(x_test, y_test)
    return model

