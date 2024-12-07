import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_generator(image_array, labels, batch_size=32, input_shape=(28,28,3)):
	# Use a sequential model for data augmentation

	data_augmentation = tf.keras.Sequential([
		tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images
		tf.keras.layers.RandomRotation(0.2),  # Rotate images by up to 20%
		tf.keras.layers.RandomContrast(0.2)  # Adjust contrast by up to 20%
		])

	dataset = tf.data.Dataset.from_tensor_slices((image_array, labels))
	dataset = dataset.shuffle(buffer_size=len(image_array)).batch(batch_size)

	# Apply data augmentation
	def augment(image, label):
		image = tf.image.convert_image_dtype(image, tf.float32)
		aug_image = data_augmentation(image)
		return aug_image, label

	dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	
	return dataset

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

# Fake data creation
def create_fake_data(num_samples, input_shape, num_classes):
	image_arrays = np.random.randint(0, 256, size=(num_samples, *input_shape), dtype=np.uint8)
	
	random_labels = np.random.randint(0, num_classes, size=num_samples)
	labels = tf.keras.utils.to_categorical(random_labels, num_classes)
	
	return image_arrays, labels