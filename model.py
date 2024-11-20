import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def plant_cnn(input_shape, num_classes, num_hidden_layers, dropout_rate=0.5):
	inputs = tf.keras.Input(shape=input_shape)
	
	x = inputs
	for _ in range(num_hidden_layers):
		x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Dropout(dropout_rate)(x)
	
	x = layers.Flatten()(x)
	x = layers.Dense(64, activation='relu')(x)
	x = layers.Dropout(dropout_rate)(x)
	outputs = layers.Dense(num_classes, activation='softmax')(x)
	
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model