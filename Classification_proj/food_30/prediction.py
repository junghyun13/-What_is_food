from tqdm import tqdm
import tensorflow as tf
import numpy as np
from glob import glob
from utils import FoodDataPaths

np.random.seed(123)

class Prediction(FoodDataPaths):

    def _load_image(self, path, img_size):
        # Read and decode the image
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        
        # Resize the image to 224x224 to match the model's expected input size
        img = tf.image.resize(img, (224, 224))  # Resizing the image to 224x224

        # Extract label from the file path (adjust this based on your dataset structure)
        label = int(path.split('/')[-2].split('_')[-1])
        
        return img, label

    def predict_test(self, model, img_size, nums=0):
        # Get the test image paths
        test_image_path = glob(self.test_image_path)
        correct = 0
        
        # If nums is specified, choose a random subset of the test images
        if nums == 0:
            img_paths = test_image_path
        else:
            img_paths = np.random.choice(test_image_path, nums, replace=False)

        # Iterate over the test images
        for img_path in tqdm(img_paths):
            img, label = self._load_image(img_path, img_size)
            
            # Add an extra dimension to the image to match the model's expected input shape
            img = np.expand_dims(img, axis=0)  # Now the shape will be (1, 224, 224, 3)

            # Predict the label of the image
            pred = int(np.argmax(model.predict(img)))  # Get the class with the highest probability

            # Check if the prediction matches the label
            if pred == label:
                correct += 1

        # Print the accuracy
        accuracy = round(correct / len(img_paths), 4) * 100
        print(f'정확도: {accuracy}%')
