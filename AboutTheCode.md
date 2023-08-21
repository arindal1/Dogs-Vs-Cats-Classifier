# Breaking Down the Code

## What is Transfer Leaning?

> Transfer learning is a machine learning technique where a model trained on one task or dataset is used as a starting point for training a model on a different, but related, task or dataset. Instead of training a model from scratch, transfer learning leverages the knowledge and features learned from the source task or dataset to improve the learning process for the target task.
>>The general idea behind transfer learning is that the features learned by a model while solving one problem can be beneficial for solving a related problem, even if the two problems are not exactly the same. This approach can be particularly useful when the target dataset is small or lacks sufficient labeled examples, as the pre-trained model can provide a foundation of general knowledge that helps the model adapt and generalize better to the new task.

In transfer learning, there are typically two main stages:

1. **Pre-training:** A model is trained on a large dataset and a relevant task, often referred to as the source task. This helps the model learn general features, patterns, and representations from the data.
2. **Fine-tuning:** After pre-training, the model is further trained on the target task using a smaller dataset. The model's weights are adjusted and fine-tuned to adapt its knowledge to the specifics of the target task. This step allows the model to specialize and improve its performance on the new task.

## Steps

The different steps to build this predictive model:

### Data Collection

1. Download the dataset from the "Dogs vs Cats" Kaggle competition.
2. Extract the downloaded dataset and organize the images into the appropriate folders.

### Preprocessing

1. Resize the images to a consistent size (e.g., 224x224 pixels).
2. Convert the images to RGB format and preprocess them for model input.
3. Split the dataset into training and testing sets.

## Model Building

### Loading Pretrained Model

We leverage the MobileNet V2 model from TensorFlow Hub as a feature extractor.

### Creating the Model

1. Define a sequential model using TensorFlow's `Sequential` API.
2. Add the pretrained MobileNet V2 model as the first layer.
3. Add a dense layer for classification.

### Model Compilation

1. Compile the model using the appropriate optimizer and loss function.
2. Specify evaluation metrics.

### Model Training

1. Train the model using the training dataset.
2. Monitor training progress and save the best model checkpoint.

## Model Evaluation

1. Evaluate the model's performance using the testing dataset.
2. Calculate accuracy and other relevant metrics.

## Making Predictions

1. Load the trained model.
2. Preprocess an input image for prediction.
3. Use the model to predict whether the image contains a dog or a cat.

## About MobileNet V2

MobileNetV2 is a deep learning architecture designed for efficient and lightweight image classification and feature extraction tasks, particularly suited for mobile and embedded devices with limited computational resources. It is a successor to the original MobileNetV1 architecture and is known for its high efficiency and accuracy trade-off.

Key features of MobileNetV2 include:

1. **Depthwise Separable Convolutions:** MobileNetV2 employs depthwise separable convolutions, which decompose the standard convolution operation into two separate layers: depthwise convolution and pointwise convolution. This significantly reduces computational complexity while maintaining good accuracy.

2. **Inverted Residuals:** MobileNetV2 introduces inverted residuals, which involve using a bottleneck architecture with a narrow pointwise convolution followed by a wider pointwise convolution. This reduces the number of parameters and computations while enhancing the model's ability to learn.

3. **Linear Bottlenecks:** The bottlenecks in MobileNetV2 use linear activation functions instead of non-linear activation functions like ReLU. This helps in retaining more information in the network and improving the model's efficiency.

4. **Multiple Scales:** MobileNetV2 has built-in support for multiple input sizes, allowing the model to adapt to various image resolutions without retraining. This is particularly useful for applications where images of different sizes need to be processed.

5. **Efficient Network Design:** MobileNetV2 focuses on striking a balance between model size, speed, and accuracy. This design philosophy makes it well-suited for real-time or resource-constrained scenarios.

## Lets understand the Code



```python
# Install required packages
!pip install kaggle
```
This installs the `kaggle` package, which is used for interacting with the Kaggle platform.

```python
# Create the .kaggle directory and copy API key
import os

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

api_key_content = '{"username":"your_username","key":"your_APIKEY"}'  # replace these values with the values in your kaggle.json
api_key_path = os.path.join(kaggle_dir, "kaggle.json")
with open(api_key_path, "w") as api_key_file:
    api_key_file.write(api_key_content)

os.chmod(api_key_path, 0o600)
```
This code sets up Kaggle API authentication by creating a `.kaggle` directory in your home folder and copying the Kaggle API key content into a `kaggle.json` file. It then sets appropriate permissions for the file.

```python
# Download dataset from Kaggle competition
!kaggle competitions download -c dogs-vs-cats
```
This command uses the Kaggle API to download the dataset for the "Dogs vs. Cats" competition.

```python
# Extract the downloaded datasets
from zipfile import ZipFile

dataset = '/content/dogs-vs-cats.zip'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('The dataset is extracted')
```
This code extracts the downloaded dataset archive.

```python
# Extract the training images
dataset = '/content/train.zip'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('The dataset is extracted')
```
This code extracts the training images from another archive.

```python
# Count the number of images in the 'train' folder
import os

path, dirs, files = next(os.walk('/content/train'))
file_count = len(files)
print('Number of images: ', file_count)
```
This code uses the `os.walk()` function to count the number of images in the 'train' folder and prints the count.

```python
# Display a few file names from the 'train' folder
file_names = os.listdir('/content/train/')
print(file_names)
```
This code lists and prints the names of a few files in the 'train' folder.

```python
# Preprocess the image data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import cv2

# Display a dog image
img = mpimg.imread('/content/train/dog.8298.jpg')
imgplt = plt.imshow(img)
plt.show()

# Display a cat image
img = mpimg.imread('/content/train/cat.4352.jpg')
imgplt = plt.imshow(img)
plt.show()
```
This section involves image preprocessing steps, including displaying sample images using `matplotlib` and `PIL`.

```python
# Create lists of file names and labels
file_names = os.listdir('/content/train/')
labels = []

for i in range(2000):
    file_name = file_names[i]
    label = file_name[0:3]

    if label == 'dog':
        labels.append(1)
    else:
        labels.append(0)

print(filenames[0:5])
print(len(filenames))
print(labels[0:5])
print(len(labels))
```
This section prepares a list of file names and corresponding labels. It extracts the labels from the file names by checking if they start with 'dog' or 'cat'.

```python
# Count the number of dog and cat images
values, counts = np.unique(labels, return_counts=True)
print(values)
print(counts)
```
This code counts the number of dog and cat images among the first 2000 images.

```python
# Load and preprocess images using OpenCV
import cv2
import glob

image_directory = '/content/image resized/'
image_extension = ['png', 'jpg']

files = []

[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extension]

dog_cat_images = np.asarray([cv2.imread(file) for file in files])

print(dog_cat_images.shape)
```
This code uses OpenCV and glob to load and preprocess images by reading them as NumPy arrays. The images are loaded into a NumPy array called `dog_cat_images`.

```python
# Split the data into training and testing sets
X = dog_cat_images
Y = np.asarray(labels)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
```
This code splits the image data and labels into training and testing sets using the `train_test_split` function from `sklearn`. It also prints the shapes of the arrays.

```python
# Scale the image data
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255
```
This code scales the image data to values between 0 and 1.

```python
# Import TensorFlow and TensorFlow Hub
import tensorflow as tf
import tensorflow_hub as hub

# Load the MobileNet model from TensorFlow Hub
mobilenet_model = '

