# Titanic Survival Prediction using Deep Learning

## Project Overview

This project focuses on predicting passenger survival on the Titanic using Deep Learning and Machine Learning techniques. The model analyzes passenger information such as age, gender, ticket details, passenger class, fare, cabin information, and embarkation port to determine whether a passenger survived the disaster.

The project includes data exploration, missing value handling, feature preprocessing, label encoding, neural network development, model evaluation, and submission file generation.

> **Note:** This project was completed under the supervision of the teacher during a live class lecture.

---

## Problem Statement

The sinking of the Titanic is one of the most well-known maritime disasters in history. The objective of this project is to predict whether a passenger survived or not based on passenger information.

This is a binary classification problem where the model learns patterns from historical passenger data and predicts survival outcomes.

---

## Target Variable

| Column | Description |
|----------|-------------|
| Survived | Survival status of the passenger |

### Target Classes

```text
0 = Did Not Survive
1 = Survived
```

---

## Dataset Description

The dataset contains demographic and travel information for Titanic passengers.

### Features

| Feature | Description |
|----------|-------------|
| PassengerId | Unique passenger identifier |
| Pclass | Passenger class |
| Name | Passenger name |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Ticket fare |
| Cabin | Cabin information |
| Embarked | Port of embarkation |
| Survived | Target variable |

---

## Project Workflow

### 1. Import Required Libraries

The project uses the following libraries:

```python
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
tensorflow
```

These libraries are used for data manipulation, visualization, preprocessing, model building, and evaluation.

---

### 2. Data Loading

The following files are loaded:

```text
dataset/train.csv
dataset/test.csv
dataset/gender_submission.csv
```

The training dataset is used for model development, while the test dataset is used to generate final predictions.

---

### 3. Exploratory Data Analysis (EDA)

The dataset is explored using:

- Dataset information
- Data types
- Missing value analysis
- Statistical summaries
- Feature inspection

Visualizations include:

- Missing value heatmaps
- Data distribution analysis
- Feature exploration

---

### 4. Missing Value Handling

One of the major objectives of this project is handling missing data intelligently.

The following columns contained missing values:

```text
Age
Cabin
Embarked
```

To solve this problem, Random Forest models are used for missing value prediction.

### Approach Used

#### Categorical Features

For categorical columns:

```python
RandomForestClassifier
```

is used to predict missing values.

#### Numerical Features

For numerical columns:

```python
RandomForestRegressor
```

is used to estimate missing values.

This approach helps preserve data patterns better than simple mean or mode imputation.

---

### 5. Data Cleaning

The dataset is cleaned before model training.

Operations performed:

- Missing value imputation
- Label encoding
- Feature selection
- Removal of unnecessary columns

Dropped columns include:

```text
Name
Ticket
Cabin
```

These columns were removed because they contain high-cardinality text data that may not contribute effectively to the neural network model.

---

### 6. Feature Encoding

Categorical features are converted into numerical format using Label Encoding.

Encoded columns include:

```text
Sex
Embarked
```

and other categorical features present in the dataset.

---

### 7. Train-Test Split

The dataset is divided into:

```text
80% Training Data
20% Validation Data
```

This allows the model to be evaluated on unseen data.

---

## Deep Learning Model

A fully connected Deep Neural Network (DNN) is built using TensorFlow.

### Network Architecture

```text
Input Layer

Dense Layer: 128 neurons, ReLU activation
Dense Layer: 64 neurons, ReLU activation
Dense Layer: 32 neurons, ReLU activation
Dense Layer: 16 neurons, ReLU activation
Dense Layer: 8 neurons, ReLU activation

Output Layer: 1 neuron, Sigmoid activation
```

---

## Model Compilation

The model is compiled using:

```text
Optimizer: Adam
Loss Function: Binary Crossentropy
Metric: Accuracy
```

---

## Early Stopping

To prevent overfitting, Early Stopping is used.

```text
Patience = 15
```

Training automatically stops when validation performance no longer improves.

---

## Model Training

Training configuration:

```text
Epochs = 100
Batch Size = 30
Validation Split = Test Dataset
```

During training, the following metrics are monitored:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

---

## Model Evaluation

The model is evaluated using:

### Accuracy

Measures the percentage of correctly classified passengers.

### Binary Crossentropy Loss

Measures prediction error during training.

### Confusion Matrix

A confusion matrix is generated to visualize:

```text
True Positives
True Negatives
False Positives
False Negatives
```

This helps understand model performance in greater detail.

---

## Prediction Generation

After training, predictions are generated on the test dataset.

### Preprocessing Steps

Before prediction:

- Encode categorical variables
- Remove unnecessary columns
- Handle remaining missing values

### Submission File

Predictions are saved into:

```text
submission.csv
```

The final submission file contains:

```text
PassengerId
Survived
```

---

## Project Structure

```text
Titanic_through_Deep_learning/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── titanic_using_deeplearning.ipynb
├── requirements.txt
├── README.md
└── submission.csv
```

---

## Requirements

Create a `requirements.txt` file and add:

```txt
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
tensorflow
jupyter
notebook
```

---

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 3: Run the Notebook

Open:

```text
titanic_using_deeplearning.ipynb
```

Run all cells from top to bottom.

---

## Final Output

The project generates:

```text
submission.csv
```

which contains survival predictions for passengers in the test dataset.

---

## Key Learning Outcomes

This project demonstrates:

- Exploratory Data Analysis (EDA)
- Missing Value Prediction using Random Forest
- Data Cleaning and Feature Engineering
- Label Encoding
- Deep Neural Networks
- Binary Classification
- Early Stopping
- Confusion Matrix Analysis
- Survival Prediction
- Kaggle-style Submission Generation

---

## Future Improvements

Possible improvements include:

- Feature Scaling
- Hyperparameter Tuning
- Cross Validation
- Ensemble Learning
- XGBoost Comparison
- Feature Importance Analysis
- ROC-AUC Evaluation
- Streamlit Deployment
- FastAPI Deployment

---

## Author

**Saif Ullah**

Artificial Intelligence Graduate

Areas of Interest:

- Machine Learning
- Deep Learning
- Data Science
- Computer Vision
- Generative AI

---

## Acknowledgment

This project was completed under the supervision of the teacher during a live classroom lecture and is intended for educational and learning purposes.

---

## License

This project is intended for educational, research, and portfolio purposes.