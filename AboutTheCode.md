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

Absolutely, I'll explain each step of the provided code in detail:

1. **Installing Kaggle Library**:

```python
!pip install kaggle
```

This command installs the Kaggle library, allowing you to interact with the Kaggle platform programmatically.

2. **Creating Kaggle API Key**:

The code sets up your Kaggle API key by writing your Kaggle username and API key to a `kaggle.json` file in the `.kaggle` directory. This step is essential for authenticating and downloading datasets from Kaggle.

3. **Downloading Competition Dataset**:

```python
!kaggle competitions download -c dogs-vs-cats
```

This command uses the Kaggle API to download the dataset for the "Dogs vs. Cats" competition. The dataset is downloaded as a zip file.

4. **Extracting Dataset**:

```python
from zipfile import ZipFile

dataset = '/content/dogs-vs-cats.zip'

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')
```

This code block extracts the downloaded zip file containing the competition dataset. It's similar to unzipping the contents of the file.

5. **Preparing Training Data**:

```python
original_folder = '/content/train/'
resized_folder = '/content/image_resized/'
# ... (further code for resizing images and preparing labels)
```

These lines of code handle preparing the training data. Images are resized to a consistent size (224x224) and converted to RGB format. Labels are assigned based on the filenames.

6. **Loading and Preprocessing Images**:

```python
image_directory = '/content/image_resized/'
image_extension = ['png', 'jpg']

files = []
[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extension]

dog_cat_images = np.asarray([cv2.imread(file) for file in files])
```

This part of the code loads and preprocesses the resized images. It creates a list of image files, then uses OpenCV (`cv2`) to read and store these images as a NumPy array (`dog_cat_images`).

7. **Splitting Data and Scaling**:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# ...
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255
```

The data is split into training and testing sets, and the pixel values are scaled to the range of [0, 1] by dividing them by 255.

8. **Creating and Training the Model**:

```python
# ...
mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
# ...
model.fit(X_train_scaled, Y_train, epochs=5)
```

A pre-trained MobileNetV2 model is loaded using TensorFlow Hub. The model is compiled and trained using the scaled training data.

9. **Evaluating the Model**:

```python
score, acc = model.evaluate(X_test_scaled, Y_test)
print('Test Loss =', score)
print('Test Accuracy =', acc)
```

The model's performance is evaluated on the test set, and the test loss and accuracy are printed.

10. **Image Prediction**:

```python
input_image_path = input('Path of the image to be predicted: ')
# ...
input_prediction = model.predict(image_reshaped)
# ...
```

The user provides the path to an image for prediction. The input image is loaded, resized, scaled, and passed through the model for prediction. The predicted label and confidence scores are printed.

## Error Handling

Please note that the above code is most suitable for a Google collab environment. Things like **'cv2_imshow'** may not work in other environments.
Here are few alternatives of code snippets for Jupyter Notebook environment:



```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
This code snippet works best for Unix and Google Collab.

For Notebook and Windows, use:

```python
import os

# Create the .kaggle directory
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Copy the kaggle.json file to the .kaggle directory
api_key_content = '{"username":"you_username","key":"your_APIKEY"}'  # Replace with your actual API key from kaggle.json
api_key_path = os.path.join(kaggle_dir, "kaggle.json")
with open(api_key_path, "w") as api_key_file:
    api_key_file.write(api_key_content)

# Set appropriate permissions
os.chmod(api_key_path, 0o600)
```




```python
!ls
```
This command is not that important, it just print the current directory and works for Unix.

```python
!dir
```



You might encounter an error in the **Predictive Model** if you are using Jupyer Notebook.

```python
input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

input_image_resize = cv2.resize(input_image, (224,224))

input_image_scaled = input_image_resize/255

image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

input_prediction = model.predict(image_reshaped)

print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label == 0:
  print('The image if of a Cat')

else:
  print('The image is of a Dog')
```

You'll encountered an error because the **cv2_imshow** function is specific to Google Collab environment, and it's not available in regular Python environments. To display images in a regular Python environment, you can use matplotlib.pyplot.imshow instead.

Here's how you can modify the code for displaying the input image:

```python
input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

# cv2_imshow(input_image)
# cv2_imshow is specific to Google Collab environment

plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)) # we display the image using matplotlib function

input_image_resize = cv2.resize(input_image, (224,224))

input_image_scaled = input_image_resize/255

image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

input_prediction = model.predict(image_reshaped)

print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label == 0:
  print('The image if of a Cat')

else:
  print('The image is of a Dog')
```
