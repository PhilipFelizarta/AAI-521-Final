from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_dataset(directory):
	file_paths = []
	labels = []

	for label_dir in os.listdir(directory):
		# Full path to the label directory
		label_path = os.path.join(directory, label_dir)

		# Skip if not a directory
		if not os.path.isdir(label_path):
			continue

		# Collect all .jpg files in the directory
		for file in os.listdir(label_path):
			if file.lower().endswith('.jpg'):  # Filter .jpg files
				file_paths.append(os.path.join(label_path, file))
				labels.append(label_dir)  # Folder name is the label

	# Create a DataFrame for better organization
	dataset = pd.DataFrame({
		"file_path": file_paths,
		"label": labels
	})
	
	return dataset

def load_image_with_pillow(image_path, target_size=(128, 128)):

	# Open the image file
	img = Image.open(image_path)
	# Resize the image
	img = img.resize(target_size)
	img = img.convert('RGB')
	img_array = np.array(img)
	return img_array

def load_images_and_labels(dataframe, target_size=(128, 128)):
	image_arrays = []
	labels = []

	for _, row in dataframe.iterrows():
		# Load the image
		img_array = load_image_with_pillow(row['file_path'], target_size=target_size)
		image_arrays.append(img_array)
		# Append the label
		labels.append(row['label'])

	# Convert to NumPy arrays
	return np.array(image_arrays), np.array(labels)

def create_numpy_dataset(directory, label_encoder=None, target_size=(128, 128)):
	# Prepare the dataset
	df = prepare_dataset(directory)

	# Load images and labels
	examples, labels = load_images_and_labels(df, target_size=target_size)

	# If no label encoder is provided, create and fit one
	if label_encoder is None:
		label_encoder = LabelEncoder()
		label_encoder.fit(labels)  # Fit on the labels of the training dataset

	# Encode labels using the provided or newly created label_encoder
	encoded_labels = label_encoder.transform(labels)

	return examples, encoded_labels, label_encoder