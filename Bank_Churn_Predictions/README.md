# Bank Customer Churn Prediction

## Project Overview

This project predicts whether a bank customer will stay with the bank or leave based on their transaction history and personal banking details. The main goal is to help banks identify customers who are likely to churn so they can take action to improve customer retention.

The project includes data loading, exploratory data analysis, feature preprocessing, deep learning model training, XGBoost classification, model evaluation, and final submission file generation.

---

## Problem Statement

Banks lose revenue when existing customers leave. By analyzing customer information such as credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, active membership status, and estimated salary, this project predicts whether a customer is likely to churn.

---

## Target Variable

| Column | Description |
|---|---|
| Exited | Whether the customer has churned or not |

### Target Classes

| Value | Meaning |
|---|---|
| 0 | Customer stays |
| 1 | Customer leaves |

---

## Dataset Description

| Feature | Description |
|---|---|
| CustomerId | Unique identifier for each customer |
| Surname | Customer's surname or last name |
| CreditScore | Customer's credit score |
| Geography | Country where the customer resides |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years the customer has been with the bank |
| Balance | Customer's account balance |
| NumOfProducts | Number of bank products used by the customer |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Estimated salary of the customer |
| Exited | Target variable showing whether the customer churned |

---

## Project Workflow

### 1. Import Required Libraries

The project starts by importing important Python libraries used for data analysis, visualization, preprocessing, model building, and evaluation.

Libraries used include:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- XGBoost

---

### 2. Load Dataset

The following files are loaded:

```text
Dataset/train.csv
Dataset/test.csv
Dataset/sample_submission.csv
```

The training dataset is used for model training and validation, while the test dataset is used to generate final churn probability predictions.

---

### 3. Data Exploration

Basic dataset exploration is performed to understand the structure of the data.

The following checks are performed:

- Display first few records
- Check dataset shape
- Check column names
- View data types
- Check missing values
- Generate statistical summary

---

### 4. Exploratory Data Analysis

Exploratory Data Analysis is performed on both numerical and categorical columns.

#### Numerical Columns

```text
CreditScore
Age
Balance
EstimatedSalary
```

For these columns, distribution plots are created with mean and median lines.

#### Categorical Columns

```text
Geography
Gender
Tenure
NumOfProducts
HasCrCard
IsActiveMember
Exited
```

Count plots are created to understand the distribution of each categorical feature.

---

### 5. Data Preprocessing

Categorical columns are converted into numerical format using Label Encoding.

Encoded columns:

```text
Geography
Gender
```

Unnecessary columns are removed before training:

```text
id
CustomerId
Surname
Exited
```

The final feature matrix and target variable are prepared as:

```python
X = df.drop(['Exited', 'id', 'CustomerId', 'Surname'], axis=1)
y = df['Exited']
```

---

### 6. Train-Test Split

The dataset is divided into training and testing sets using an 80:20 split.

```python
test_size = 0.2
random_state = 42
```

This allows the model to be trained on one part of the data and evaluated on unseen validation data.

---

## Models Used

### 1. Deep Neural Network

A deep neural network is built using TensorFlow/Keras for binary classification.

#### Architecture

```text
Input Layer
Dense Layer: 64 neurons, ReLU activation
Dense Layer: 32 neurons, ReLU activation
Dense Layer: 16 neurons, ReLU activation
Dense Layer: 8 neurons, ReLU activation
Output Layer: 1 neuron, Sigmoid activation
```

#### Compilation

```text
Optimizer: Adam
Learning Rate: 0.001
Loss Function: Binary Crossentropy
Metric: Accuracy
```

#### Early Stopping

Early stopping is used to prevent overfitting. Training stops when the validation performance does not improve for several epochs.

---

### 2. XGBoost Classifier

An XGBoost classifier is also trained for customer churn prediction.

#### Model Parameters

```text
n_estimators = 500
learning_rate = 0.1
max_depth = 6
random_state = 42
```

XGBoost is used because it performs well on structured tabular data and is effective for classification problems.

---

## Model Evaluation

The XGBoost model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

A confusion matrix is also plotted to compare actual and predicted churn values.

---

## Prediction and Submission

After training, the XGBoost model predicts churn probabilities on the test dataset.

The probability of class `1`, meaning customer churn, is saved in the submission file.

```python
submission['Exited'] = y_pred[:, 1]
submission.to_csv('submission.csv', index=False)
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
xgboost
jupyter
notebook
```

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/isaifullah/bank-churn-prediction.git
cd bank_churn_prediction
```

### Step 2: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 3: Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 4: Run the Notebook

Open the notebook file:

```text
bank_churn_project.ipynb
```

Run all cells from top to bottom.

---

## Final Output

The final output of this project is:

```text
submission.csv
```

This file contains predicted churn probabilities for customers in the test dataset.

---

## Key Learning Outcomes

This project demonstrates:

- Customer churn prediction
- Data preprocessing
- Exploratory data analysis
- Label encoding
- Binary classification
- Deep neural network modeling
- XGBoost classification
- Model evaluation
- Submission file generation

---

## Future Improvements

The project can be improved further by:

- Applying feature scaling before neural network training
- Using One-Hot Encoding instead of Label Encoding for categorical features
- Performing hyperparameter tuning
- Comparing multiple machine learning models
- Using ROC-AUC score for better evaluation
- Adding feature importance analysis
- Deploying the model using Streamlit or FastAPI

---

## Author

**Saif Ullah**

Artificial Intelligence Graduate  
Interested in Machine Learning, Deep Learning, Data Science, Computer Vision, and Generative AI.

---

## License

This project is created for educational, learning, and portfolio purposes.