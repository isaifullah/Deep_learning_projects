# Rice Disease Classification using CNN with TensorFlow

## Project Overview

This project focuses on classifying rice plant diseases using a Convolutional Neural Network (CNN) built with TensorFlow. The model is trained on rice leaf images and learns to identify different rice disease categories based on visual patterns in the images.

The project includes dataset exploration, image visualization, label analysis, image preprocessing, CNN model building, model training, evaluation, and visualization of training performance.

> **Note:** This project was completed under the supervision of the teacher during a live class lecture.

---

## Problem Statement

Rice diseases can reduce crop quality and production if they are not detected early. Manual disease identification requires expert knowledge and can be time-consuming. This project uses deep learning to classify rice plant diseases from images, helping demonstrate how computer vision can be applied in agriculture.

---

## Objective

The main objective of this project is to build a CNN-based image classification model that can classify rice plant images into different disease categories.

---

## Dataset Description

The dataset used in this project is the Paddy Disease Classification dataset.

The dataset contains rice plant images organized into disease-specific folders. It also includes a `train.csv` file containing image metadata.

### Metadata Columns

| Column | Description |
|---|---|
| image_id | Name or ID of the image |
| label | Disease class of the rice plant |
| variety | Rice variety name |
| age | Age of the rice plant in days |

---

## Disease Classes

The dataset contains 10 unique rice disease classes:

```text
bacterial_leaf_blight
bacterial_leaf_streak
bacterial_panicle_blight
blast
brown_spot
dead_heart
downy_mildew
hispa
normal
tungro
```

---

## Dataset Observations

Based on the analysis performed in the notebook:

- The training dataset contains 10,407 images.
- There are 10 unique rice disease classes.
- There are 10 rice varieties in the dataset.
- Rice plant age ranges from 45 to 85 days.
- ADT45 is the most common rice variety in the training data.

---

## Project Workflow

### 1. Import Required Libraries

The project uses Python libraries for data analysis, visualization, preprocessing, and deep learning.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
```

---

### 2. Load Dataset

The metadata file is loaded using Pandas.

```python
data = pd.read_csv("dataset/paddy_disease_classification/train.csv")
```

Image data is loaded from:

```text
dataset/paddy_disease_classification/train_images
```

---

### 3. Data Exploration

The notebook explores the dataset by checking:

- First few records
- Dataset shape
- Unique disease labels
- Unique rice varieties
- Age distribution
- Disease distribution
- Variety distribution

---

### 4. Data Visualization

Several visualizations are created to better understand the dataset.

#### Variety Distribution

A histogram is plotted to show the distribution of different rice varieties.

#### Disease Distribution

A histogram is plotted to show the count of images for each disease class.

#### Sample Image Visualization

The notebook displays sample images from:

- Normal rice plants
- Dead heart disease
- All available disease classes

This helps visually understand the differences between disease categories.

---

### 5. Label Encoding

The `label` and `variety` columns are encoded using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
data['variety'] = label_encoder.fit_transform(data['variety'])
```

This converts categorical text labels into numerical values.

---

### 6. Image Dataset Loading

The images are loaded using TensorFlow's `image_dataset_from_directory`.

```python
tf.keras.utils.image_dataset_from_directory()
```

The dataset is split into:

```text
Training set: 80%
Validation set: 20%
```

### Image Parameters

```text
Image Height: 224
Image Width: 224
Batch Size: 32
Seed: 123
```

---

### 7. Image Normalization

Images are normalized by scaling pixel values from 0–255 to 0–1.

```python
tf.keras.layers.Rescaling(1./255)
```

Normalization helps the CNN train more efficiently.

---

### 8. Dataset Optimization

The TensorFlow data pipeline is optimized using:

```python
cache()
prefetch()
```

This improves training performance by preparing the next batch while the model is training.

---

## CNN Model Architecture

The CNN model is built using TensorFlow Sequential API.

```text
Rescaling Layer

Conv2D Layer: 128 filters, kernel size 3, ReLU activation
MaxPooling2D Layer

Conv2D Layer: 64 filters, kernel size 3, ReLU activation
MaxPooling2D Layer

Conv2D Layer: 32 filters, kernel size 3, ReLU activation
MaxPooling2D Layer

Conv2D Layer: 16 filters, kernel size 3, ReLU activation
MaxPooling2D Layer

Flatten Layer
Dropout Layer: 0.25
Dense Layer: 128 neurons, ReLU activation
Output Layer: 10 neurons, Softmax activation
```

---

## Model Compilation

The model is compiled using:

```text
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metric: Accuracy
```

---

## Model Training

The model is trained using the training dataset and validated using the validation dataset.

```text
Epochs: 10
Callback: EarlyStopping
EarlyStopping Patience: 30
```

EarlyStopping is used to stop training when validation performance stops improving.

---

## Model Evaluation

The model is evaluated on the validation dataset.

The following results are monitored:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

The notebook also plots:

- Loss curve
- Accuracy curve

These plots help understand model learning behavior and possible overfitting.

---

## Project Structure

```text
Rice_Disease_Classification/
│
├── dataset/
│   └── paddy_disease_classification/
│       ├── train.csv
│       └── train_images/
│           ├── bacterial_leaf_blight/
│           ├── bacterial_leaf_streak/
│           ├── bacterial_panicle_blight/
│           ├── blast/
│           ├── brown_spot/
│           ├── dead_heart/
│           ├── downy_mildew/
│           ├── hispa/
│           ├── normal/
│           └── tungro/
│
├── rice_disease_classification.ipynb
├── requirements.txt
└── README.md
```

---

## Requirements

Create a `requirements.txt` file and add the following libraries:

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
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

### Step 3: Run the Notebook

Open the notebook file:

```text
05_rice_disease_classification.ipynb
```

Run all cells from top to bottom.

---

## Final Output

The final output of this project is a trained CNN model that classifies rice plant images into one of the 10 disease categories.

The model can identify classes such as:

- Normal
- Dead heart
- Blast
- Brown spot
- Hispa
- Tungro
- Bacterial leaf blight
- Bacterial leaf streak
- Bacterial panicle blight
- Downy mildew

---

## Key Learning Outcomes

This project demonstrates:

- Image classification using CNN
- Agricultural disease classification
- TensorFlow image dataset loading
- Image preprocessing and normalization
- Dataset visualization
- Label encoding
- CNN architecture design
- Model training and validation
- Accuracy and loss visualization
- Use of cache and prefetch for performance optimization

---

## Future Improvements

This project can be improved by:

- Saving the trained model for future use
- Adding test image prediction functionality
- Using transfer learning models such as MobileNetV2, EfficientNet, or ResNet
- Applying data augmentation
- Adding confusion matrix and classification report
- Improving class imbalance handling
- Deploying the model using Streamlit
- Displaying disease descriptions and treatment suggestions

---

## Author

**Saif Ullah**

Artificial Intelligence Graduate  
Interested in Machine Learning, Deep Learning, Computer Vision, Data Science, and Generative AI.

---

## License

This project is created for educational, learning, and portfolio purposes.