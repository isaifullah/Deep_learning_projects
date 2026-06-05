# CNN-Based MNIST and Fashion MNIST Image Classification

## Project Overview

This project contains two Convolutional Neural Network (CNN) based image classification notebooks developed using TensorFlow and Keras.

The first notebook classifies handwritten digits from the MNIST dataset, while the second notebook classifies clothing items from the Fashion MNIST dataset. Both projects demonstrate the basic workflow of deep learning for image recognition, including dataset loading, visualization, preprocessing, CNN model building, training, evaluation, model saving, and single-image prediction.

> **Note:** This project was completed under the supervision of the teacher during a live class lecture.

---

## Projects Included

### 1. MNIST Handwritten Digit Classification

The MNIST project focuses on classifying grayscale images of handwritten digits from 0 to 9.

#### Dataset

The MNIST dataset contains:

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Number of classes: 10

#### Classes

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### Saved Model

```text
models_saved/mnist_cnn.h5
```

---

### 2. Fashion MNIST Classification

The Fashion MNIST project focuses on classifying grayscale images of fashion products.

#### Dataset

The Fashion MNIST dataset contains:

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Number of classes: 10

#### Classes

```text
0 = T-shirt/top
1 = Trouser
2 = Pullover
3 = Dress
4 = Coat
5 = Sandal
6 = Shirt
7 = Sneaker
8 = Bag
9 = Ankle boot
```

#### Saved Model

```text
models_saved/fashion_mnist_cnn.h5
```

---

## What is a Convolutional Neural Network?

A Convolutional Neural Network, also known as CNN, is a type of deep learning model mainly used for image recognition and image processing tasks.

CNNs automatically learn important patterns from images such as edges, shapes, textures, and object parts.

### Main CNN Components Used

#### Convolutional Layer

The convolutional layer applies filters to the image and extracts important visual features.

#### Convolutional Operation

A small filter or kernel moves across the image and performs mathematical operations to produce feature maps.

#### Max Pooling Layer

The max pooling layer reduces the size of feature maps while keeping the most important information.

#### Flatten Layer

The flatten layer converts the extracted feature maps into a one-dimensional format.

#### Dense Layer

The dense layer performs final classification based on the extracted image features.

#### Softmax Output Layer

The softmax layer gives probability scores for each class and selects the most likely class.

---

## Project Workflow

### 1. Import Required Libraries

The project uses TensorFlow, Keras, Matplotlib, and EarlyStopping callback.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
```

---

### 2. Load Datasets

For digit classification:

```python
keras.datasets.mnist.load_data()
```

For fashion item classification:

```python
keras.datasets.fashion_mnist.load_data()
```

---

### 3. Data Visualization

Sample images are displayed from the training dataset to understand the image data visually.

The project shows:

- Sample training images
- Image labels
- Grayscale image format
- Pixel value matrix

---

### 4. Data Exploration

The dataset shape and image size are checked.

Each dataset contains:

```text
Training images: 60,000
Testing images: 10,000
Image size: 28 × 28 pixels
```

---

### 5. Data Normalization

Pixel values are normalized from the range 0–255 to 0–1.

```python
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
```

Normalization helps the model train faster and improves performance.

---

## CNN Model Architecture

Both projects use the same CNN architecture.

```text
Input Layer: 28 × 28 × 1

Conv2D Layer: 10 filters, 3 × 3 kernel, ReLU activation
Conv2D Layer: 10 filters, 3 × 3 kernel, ReLU activation
MaxPooling2D Layer

Conv2D Layer: 10 filters, 3 × 3 kernel, ReLU activation
Conv2D Layer: 10 filters, 3 × 3 kernel, ReLU activation
MaxPooling2D Layer

Flatten Layer
Dense Output Layer: 10 neurons, Softmax activation
```

---

## Model Compilation

The model is compiled using:

```text
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Evaluation Metric: Accuracy
```

For the MNIST digit project, labels are converted into one-hot encoded format before training.

For the Fashion MNIST project, sparse labels are used directly.

---

## Model Training

EarlyStopping is used during training to stop the model when validation performance stops improving.

### MNIST Digit Classification

```text
Epochs: 50
Batch Size: 35
Callback: EarlyStopping
```

### Fashion MNIST Classification

```text
Epochs: 100
Batch Size: 32
Callback: EarlyStopping
```

---

## Model Evaluation

Both models are evaluated using:

- Test Loss
- Test Accuracy
- Training Loss Curve
- Validation Loss Curve
- Training Accuracy Curve
- Validation Accuracy Curve

The loss and accuracy graphs help understand how well the model learns during training.

---

## Single Image Prediction

After training, a single image is selected from the test dataset and passed to the trained CNN model.

The model predicts the class by selecting the index with the highest probability.

```python
predictions = model.predict(img.reshape(1, 28, 28, 1))
print(predictions.argmax())
```

---

## Project Structure

```text
CNN_MNIST_Datasets/
│
├── models_saved/
│   ├── mnist_cnn.h5
│   └── fashion_mnist_cnn.h5
│
├── CNN_mnist.ipynb
├── CNN_fashion_mnist.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements

Create a `requirements.txt` file and add:

```txt
tensorflow
keras
matplotlib
jupyter
notebook
```

---

## How to Run the Project

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 3: Run the Notebooks

For handwritten digit classification:

```text
CNN_mnist.ipynb
```

For fashion product classification:

```text
CNN_fashion_mnist.ipynb
```

Run each notebook cell by cell from top to bottom.

---

## Final Outputs

After running both notebooks, the trained models are saved inside the `models_saved` folder.

```text
models_saved/mnist_cnn.h5
models_saved/fashion_mnist_cnn.h5
```

---

## Key Learning Outcomes

This project helped in understanding:

- Basics of Convolutional Neural Networks
- Image classification workflow
- MNIST and Fashion MNIST datasets
- Image normalization
- CNN model building using TensorFlow/Keras
- Model training and validation
- Early stopping
- Accuracy and loss visualization
- Saving trained deep learning models
- Making predictions on unseen images

---

## Future Improvements

This project can be improved by:

- Adding dropout layers to reduce overfitting
- Increasing the number of filters
- Using batch normalization
- Adding confusion matrix evaluation
- Displaying predicted class names instead of class numbers
- Creating a Streamlit web application
- Comparing CNN performance with other deep learning models

---

## Author

**Saif Ullah**

Artificial Intelligence Graduate  
Interested in Machine Learning, Deep Learning, Computer Vision, Data Science, and Generative AI.

---

## License

This project is created for educational, learning, and portfolio purposes.