# Activity 1 : Introduction to Supervised Learning




---

> [!IMPORTANT]  
> * **Next week:** Cloud sessions will be held on **Wednesday morning, April 8th**, and **Thursday afternoon, April 9th**.
> * **The following week:** Machine Learning sessions will take place on **Wednesday morning, April 15th**, and **Thursday afternoon, April 16th**.
> This schedule is an **exceptional situation** to the usual flow, where we typically have **Cloud sessions on Wednesdays** and **Machine Learning sessions on Thursdays**.


<!-- 
> [!NOTE]  
> Throughout this lab, add text blocks in your notebook to explain any new concepts you encounter. Writing things down in your own words will help reinforce your understanding. If you're unsure about a function or concept, consider using your favorite AI assistant (like ChatGPT, Gemini, or Claude) to get more details. 
-->

---

## Part 0: Introduction and Setup

**Learning Objectives:**
By the end of this lab, you will be able to:
- Define **supervised learning** and differentiate it from other Machine Learning (ML) types.
- Identify the two main supervised learning tasks: **classification** and **regression**.
- Apply a **standard machine learning pipeline** for supervised tasks.
- Implement basic classification (Decision Tree) and regression (Linear Regression) models using `scikit-learn`.
- Evaluate model performance using appropriate metrics (accuracy, precision, recall, MAE, MSE, R²).
- Understand the necessity of **data preprocessing** for real-world datasets.


### 1. What is Supervised Learning?
Supervised learning is a type of machine learning where an algorithm learns from a **labeled dataset**. Think of it like a student learning with a teacher. The teacher provides the student with practice questions (the *input features*) along with the correct answers (the *labels* or *targets*). 

The goal is for the model to learn the underlying patterns so that when it is given *new, unseen* practice questions, it can accurately predict the answers on its own.

Supervised learning is divided into two main categories, depending on what type of answer you are trying to predict:

*   **Classification:** Predicts a discrete category or class label.
    *   *Example 1:* Is this email spam or not? (Categories: `Spam`, `Not Spam`)
    *   *Example 2:* Which species does this Iris flower belong to? (Categories: `Setosa`, `Versicolor`, `Virginica`)
*   **Regression:** Predicts a continuous numerical value.
    *   *Example 1:* What is the predicted mileage (MPG) of this car? (Value: e.g., `25.5 MPG`)
    *   *Example 2:* What will the price of this house be? (Value: e.g., `$350,000`)

> [!IMPORTANT]  
> **The Golden Rule for identifying the task:** If the answer is a category or a "yes/no", it is **Classification**. If the answer is a continuous number where "more" or "less" makes mathematical sense, it is **Regression**.


### 2. The Standard Machine Learning Pipeline
Most supervised learning projects follow a standardized workflow. Regardless of whether you are predicting spam emails or house prices, you will generally follow these 7 steps:

1.  **Load Data:** Get the data into your Python environment (typically using the `pandas` library).
2.  **Explore & Preprocess Data:** Clean the data, handle missing values, and convert text into numbers. 
3.  **Split Data:** Divide the data into a **training set** (to teach the model) and a **testing set** (to evaluate the model on data it has never seen before).
4.  **Choose & Create Model:** Select an appropriate algorithm (e.g., Decision Tree for classification, Linear Regression for regression).
5.  **Train Model:** Fit the model to the **training data**. This is where the mathematical "learning" actually happens.
6.  **Make Predictions:** Use the trained model to guess the outputs for the **testing data**.
7.  **Evaluate Model:** Compare the model's guesses against the *actual* known answers in the testing data using scoring metrics.

> [!TIP]  
> **Data Science Reality:** In the real world, **Step 2 (Preprocess Data)** takes up about 80% of a Data Scientist's time! In Parts 1-4 of this lab, we will use datasets that have *already been cleaned* for you, so you can focus on Steps 3-7. In Part 5, we will show you how raw, messy data actually looks and how it gets cleaned.


### 3. Datasets We Will Use:
We will explore four different datasets to practice both classification and regression:

*   **Titanic (Classification):** Predict passenger survival (Yes/No). *(Preprocessed for Part 1)*
*   **Iris (Classification):** Predict iris flower species based on petal measurements.
*   **Auto MPG (Regression):** Predict car fuel efficiency (Miles Per Gallon). *(Preprocessed for Part 3, Raw version in Part 5)*
*   **Diabetes (Regression):** Predict a numerical score representing disease progression.


### 4. Setup:
Open a new Google Colab notebook ([colab.research.google.com](https://colab.research.google.com)). Run the following code block in your first cell to import the necessary tools.

> [!NOTE]  
> **What do these libraries do?** 
> *   `pandas`: Acts like Excel for Python (tables and data manipulation).
> *   `matplotlib` & `seaborn`: Used for drawing charts and graphs.
> *   `sklearn` (Scikit-Learn): The industry-standard library containing the actual Machine Learning algorithms and evaluation math.

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Tools from Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 

# Configure plots for better visibility
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```



### Part 0: Knowledge Check

Before moving on to writing code, ensure you understand the core concepts. Read the questions below and try to answer them in your head before revealing the solutions.

**Question 1:** You are building an AI to look at a photo of an X-ray and determine if a bone is `Fractured` or `Not Fractured`. Is this a Classification task or a Regression task?

<details>
<summary><strong>View Answer</strong></summary>

**Classification.** 
Because the model is choosing between discrete categories ("Fractured" vs. "Not Fractured"), it is classifying the image. If the AI was trying to predict exactly *how many days* the bone would take to heal, that would be Regression.
</details>

<br>

**Question 2:** In the Machine Learning Pipeline, why do we split our data into a "Training Set" and a "Testing Set"? Why not just train the model on all the data we have?

<details>
<summary><strong>View Answer</strong></summary>

If we train the model on all the data, the model might just **memorize** the answers instead of actually learning the underlying patterns (this is called *overfitting*). 

By hiding a portion of the data (the Testing Set), we can evaluate the model like a teacher giving a final exam. If the model performs well on data it has *never seen before*, we know it has truly learned.
</details>

<br>

**Question 3:** What exactly is a "Label" (also known as a Target) in Supervised Learning? 

<details>
<summary><strong>View Answer</strong></summary>

The label is the **"correct answer"** attached to the historical data. It is the specific column or value that you are trying to train the machine learning model to predict in the future.
</details>

---

### Part 1: Guided Classification (Titanic Dataset)

**Goal:** Learn the classification pipeline by predicting passenger survival on the Titanic dataset using a Decision Tree algorithm. We will use a version of the dataset that we lightly clean directly in the code.

#### Step 1.1: Load Pre-Cleaned Data
Even when datasets are relatively "clean," they often require some basic manipulation before a machine learning model can read them. Models require numbers, not text, and they cannot handle missing blank spaces (`NaN`).

```python
# Load a Titanic dataset directly from a URL
url_titanic = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_df = pd.read_csv(url_titanic)

# Basic cleanup:
# 1. Drop columns that are unique identifiers or text that a basic model cannot process mathematically.
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, errors='ignore')

# 2. Fill missing 'Age' values with the median age of all passengers.
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# 3. Convert 'Sex' from text ('male'/'female') to numeric (0/1). This is a simple form of encoding.
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

# 4. Drop any remaining rows that still have missing values (e.g., if 'Fare' was missing).
titanic_df.dropna(inplace=True)

print("Titanic Data (first 5 rows):")
print(titanic_df.head())
print("\nData Info:")
titanic_df.info()
```

> [!TIP]  
> If you ever want to save a dataframe you have cleaned in Colab to your local computer, you can run this code in a new cell:
> `titanic_df.to_csv('my_cleaned_titanic_data.csv', index=False)`

#### Step 1.2: Define Features (X) and Target (y)
In machine learning, we separate the data into the information we are learning *from* (Features) and the answer we are trying to predict (Target). 

By convention, Features are assigned to a capital `X` (because it is a 2D table or matrix), and the Target is assigned to a lowercase `y` (because it is a 1D list or vector).

```python
# Define features (X) by dropping the target column
X_titanic = titanic_df.drop('Survived', axis=1)

# Define target (y) by selecting only the target column
y_titanic = titanic_df['Survived']

print("\nFeatures (X):")
print(X_titanic.head(2))
print("\nTarget (y):")
print(y_titanic.head(2))
```

#### Step 1.3: Split Data into Training and Testing Sets
We will use 80% of the data to train the model and hold back 20% to test it. 

```python
# Split the data
X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic = train_test_split(
    X_titanic, y_titanic, test_size=0.2, random_state=42, stratify=y_titanic 
)

print(f"\nTraining set shape: X={X_train_titanic.shape}, y={y_train_titanic.shape}")
print(f"Testing set shape: X={X_test_titanic.shape}, y={y_test_titanic.shape}")
```

> [!NOTE]  
> **Important Parameters:**
> *   `random_state=42`: Splitting data relies on randomness. Setting a specific "seed" number (like 42) ensures that every time you run this cell, the data splits the exact same way. This makes your results reproducible.
> *   `stratify=y_titanic`: Ensures that the ratio of survivors to non-survivors is exactly the same in both the training set and the testing set.

#### Step 1.4: Create and Train a Decision Tree Classifier
We initialize the model and use the `.fit()` method to train it. The `.fit()` command is where the actual mathematical learning takes place.

```python
# Create a Decision Tree Classifier model
# max_depth restricts how deep the tree can grow to prevent overfitting
clf_titanic = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the model using the training data ONLY
clf_titanic.fit(X_train_titanic, y_train_titanic)

print("\nDecision Tree Classifier model trained successfully.")
```

#### Step 1.5: Make Predictions on Test Data
Now that the model is trained, we give it the `X_test_titanic` data (the passenger details) without the answers, and ask it to predict who survived.

```python
# Make predictions
y_pred_titanic = clf_titanic.predict(X_test_titanic)

# Display first 10 predictions vs actual values
print("\nFirst 10 Predictions:", y_pred_titanic[:10])
print("First 10 Actual Values:", y_test_titanic[:10].values)
```

#### Step 1.6: Evaluate the Model
How well did our model do? We use `accuracy_score` and `classification_report` to compare the model's predictions (`y_pred_titanic`) against the actual hidden answers (`y_test_titanic`).

```python
# Evaluate the model
accuracy_titanic = accuracy_score(y_test_titanic, y_pred_titanic)
print(f"\nModel Accuracy: {accuracy_titanic:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_titanic, y_pred_titanic, target_names=['Did not Survive', 'Survived']))
```

> [!IMPORTANT]  
> **Understanding Classification Metrics:**
> *   **Accuracy:** The overall percentage of correct predictions (Total Correct / Total Guesses).
> *   **Precision:** Of all the passengers the model *claimed* survived, how many actually did? (True Positives / (True Positives + False Positives)).
> *   **Recall:** Of all the passengers who *actually* survived, how many did the model successfully find? (True Positives / (True Positives + False Negatives)).
> *   **F1-Score:** A balanced average of Precision and Recall.

#### Step 1.7: Visualize the Decision Tree (Optional)
One of the main benefits of a Decision Tree model is that it is "interpretable." We can actually print out the rules the model learned.

```python
# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(clf_titanic,
          feature_names=X_titanic.columns,
          class_names=['Did not Survive', 'Survived'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Titanic Survival Prediction")
plt.show()
```

### Knowledge Check (Part 1)

Review the concepts covered in this section. Try to answer the questions before viewing the solutions.

**Question 1: The Flaw of Accuracy**
Why is overall "Accuracy" not always a reliable metric for evaluating classification models? Provide a hypothetical example where a high accuracy score can be misleading, and explain why precision or recall would be more important.

<details>
<summary><strong>View Answer</strong></summary>

Accuracy is highly misleading when dealing with **imbalanced datasets** (where one class is much more common than the other).

**Example:** Imagine a dataset of 1,000 credit card transactions, where 990 are legitimate and only 10 are fraudulent. If you build a completely "dumb" model that simply predicts "Legitimate" for every single transaction, it will be correct 990 out of 1,000 times. That is an **Accuracy of 99%**. 

However, the model failed to catch a single fraudulent transaction. In this scenario:
*   **Recall** is much more important. We want a high Recall for fraud, meaning we successfully identify the actual fraud cases, even if it means raising a few false alarms (lower precision).
</details>

<br>

**Question 2: Confusion Matrix**
What is a confusion matrix in classification? Describe its four main components and explain how it helps assess a classifier's performance better than a single accuracy number.

<details>
<summary><strong>View Answer</strong></summary>

A confusion matrix is a table that breaks down exactly *how* a classification model is right and *how* it is wrong. It divides predictions into four categories:

1.  **True Positives (TP):** The model predicted "Yes", and the actual answer was "Yes".
2.  **True Negatives (TN):** The model predicted "No", and the actual answer was "No".
3.  **False Positives (FP):** The model predicted "Yes", but the actual answer was "No" (Type I Error / False Alarm).
4.  **False Negatives (FN):** The model predicted "No", but the actual answer was "Yes" (Type II Error / A Miss).

It is superior to a single accuracy number because it shows you exactly where the model struggles. For instance, you can look at the matrix and see, "My model is great at predicting who survived, but it is terrible at predicting who did not survive."
</details>

**Question 3: Why Precision, Recall, and F1 Score Matter in Medical Diagnosis**
In medical diagnoses, why are precision and recall not always enough, and how does the F1 score provide a better evaluation metric? Explain with an example.

<details>  
<summary><strong>View Answer</strong></summary>

In medical diagnosis, precision and recall alone may not provide a balanced view of a model’s performance, especially when there’s a need to strike a balance between false positives and false negatives. The **F1 score**, which is the harmonic mean of precision and recall, is particularly useful when both false positives and false negatives are undesirable and need to be balanced.

**Example:** Consider a medical test for detecting a rare disease that affects only 1 in 1,000 people. Suppose a diagnostic model is trained to identify whether a person has the disease (positive) or not (negative). If the model predicts "no disease" for every patient, it will have a **high accuracy** (99.9%) but will miss all actual cases of the disease. This is a clear failure of the model in a medical context, even though accuracy is high.

* **Precision** would tell us that when the model predicts someone has the disease, how likely it is that they actually do. A high precision would be important to avoid unnecessary treatments for false positives.
* **Recall** would measure how well the model identifies actual patients with the disease, minimizing the risk of false negatives (i.e., missed diagnoses).

In this case, both **Precision** and **Recall** are important, but **F1 Score** combines them into a single metric. It allows us to optimize for both detecting as many true positives (sick patients) as possible while minimizing the chance of false positives (unnecessary treatment or worry).

For instance, a model with:

* **Precision = 0.95** (95% of the positive predictions are correct)
* **Recall = 0.50** (50% of actual cases are detected)

Might have a **F1 Score** of 0.67, which is a better representation of the model's overall effectiveness than looking at either precision or recall alone.

This is why **F1 score** is crucial: it balances the importance of both **false negatives** and **false positives**, giving a more holistic view of a model’s real-world performance in scenarios like medical diagnosis.

</details> 

<br>


**Question 4: Why is Median Preferred Over Average for Filling Missing Values?**
In the context of filling missing values, why is the **median** often preferred over the **average** (mean)? What could go wrong if we used the **average** instead of the **median** in situations like filling missing 'Age' values in a dataset?

<details>  
<summary><strong>View Answer</strong></summary>

The **median** is often preferred over the **mean** (average) when filling missing values in datasets because the median is **less sensitive to outliers** and **skewed data**.

In many real-world datasets, especially those involving human characteristics (like **age**), data can be **skewed** or have **outliers**. For instance, a few extremely old passengers might significantly affect the **mean**, making it higher than the typical age for most passengers. The **median**, however, is the middle value in the sorted list of ages, so it better represents the typical value of a dataset, regardless of outliers.

**Example:**
Imagine we have the following ages for passengers on the Titanic:
`[22, 24, 30, 35, 1000]`.
The **mean** age is:
$$
\text{mean} = \frac{22 + 24 + 30 + 35 + 1000}{5} = 222.2
$$
But the **median** age is **30** (the middle value).

If we filled missing 'Age' values with the **mean**, we would be incorrectly imputing ages that are much higher than the typical value, due to the extreme outlier (1000). This would distort the dataset, especially when used in predictive models or statistical analysis.

Using the **median** ensures that the imputed value represents the **typical passenger** age rather than being skewed by extreme values. In cases where the data is reasonably symmetric and lacks outliers, using the mean could be acceptable, but in most real-world cases, the median is a safer choice for imputation.

Thus, if we used the **mean** instead of the **median** in the code:

```python
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
```

We would risk distorting the distribution of the 'Age' feature, leading to biased predictions or misleading analysis.

</details>

<br>


---

### Part 2: Practice Classification (Iris Dataset)

> [!NOTE]  
> In the original design of this lab, you were expected to write the code for this section from scratch. However, because students come from diverse programming backgrounds, we have provided the complete sample solution below. Your task is to read through the code, run it, and focus entirely on understanding the "big picture" of how the pipeline steps apply to a new dataset.

**Goal:** Apply the classification pipeline steps to a new dataset. We will use the famous "Iris" dataset, which involves predicting the specific species of an Iris flower based on the measurements of its petals and sepals.

#### Step 2.1: Load Iris Data
Unlike the Titanic dataset which we downloaded from a URL, the Iris dataset is so famous that it is built directly into the `scikit-learn` library for practice. It comes perfectly clean and entirely numerical.

```python
# Load the Iris dataset directly from sklearn
iris = load_iris()

# The sklearn dataset comes as a dictionary-like object. 
# We need to format it into a Pandas DataFrame for easier viewing.
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

# Map the numeric target (0, 1, 2) to actual species names for human readability
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Iris Data (first 5 rows):")
print(iris_df.head())
print("\nData Info:")
iris_df.info()
```

> [!TIP]  
> **What is `np.c_` doing?** 
> The `sklearn` library stores the features (the measurements) and the targets (the answers) in two separate arrays. `np.c_` is a NumPy command that literally translates to "concatenate columns". It glues the target array to the right side of the feature array so we can view them together in one single Pandas table.

#### Step 2.2: Define Features (X) and Target (y)
We want to predict the exact species of the flower. Therefore, the measurements will be our features (`X`), and the target number will be our target (`y`).

```python
# Define X_iris (features) and y_iris (target)
# We MUST drop both the 'target' column and the 'species' column to prevent the model from cheating!
X_iris = iris_df.drop(['target', 'species'], axis=1) 
y_iris = iris_df['target']

# Print shapes to verify
print(f"\nIris features shape: {X_iris.shape}")
print(f"Iris target shape: {y_iris.shape}")
```

#### Step 2.3: Split Data
We split the data into 80% for training and 20% for testing. Because the dataset is relatively small (only 150 flowers), using `stratify` is especially important to ensure every species is equally represented in the test set.

```python
# Split the data into training and testing sets
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# Print shapes to verify
print(f"\nTraining set shape: X={X_train_iris.shape}, y={y_train_iris.shape}")
print(f"Testing set shape: X={X_test_iris.shape}, y={y_test_iris.shape}")
```

#### Step 2.4: Create and Train a Decision Tree Classifier
We initialize another Decision Tree. We set `max_depth=4` this time.

```python
# Create and train the model 
clf_iris = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_iris.fit(X_train_iris, y_train_iris)

print("\nIris Decision Tree model trained.")
```

#### Step 2.5: Make Predictions
We use the trained model to predict the species for our hidden test set.

```python
# Make predictions on the unseen testing features
y_pred_iris = clf_iris.predict(X_test_iris)

# Display first 10 predictions vs actual values
print("\nFirst 10 Predictions:", y_pred_iris[:10])
print("First 10 Actual Values:", y_test_iris[:10].values)
```

#### Step 2.6: Evaluate the Model
We calculate the accuracy and print the classification report. Notice how the `target_names` parameter uses the built-in names from the dataset.

```python
# Evaluate the model
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
print(f"\nIris Model Accuracy: {accuracy_iris:.4f}")

# Print the classification report
print("\nIris Classification Report:")
print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))
```

### Knowledge Check (Part 2)

Review the concepts covered in this section. Try to answer the questions before viewing the solutions.

**Question 1: Data Leakage**
In Step 2.2, the code explicitly drops **both** the `target` column and the `species` column from the `X` (Features) dataframe. What would happen if we forgot to drop the `species` column, and passed it into the model as a feature?

<details>
<summary><strong>View Answer</strong></summary>

This would cause a massive error known as **Data Leakage**. 

The `species` column contains the exact answers the model is trying to predict (just in text format instead of numeric). If you give the model the answers as part of its training features, it will achieve 100% accuracy instantly, but it will be entirely useless in the real world when dealing with new flowers where the species is unknown. You must always ensure your target variables are strictly separated from your features.
</details>

<br>

**Question 2: Binary vs. Multi-class Classification**
In Part 1 (Titanic), the target was "Survived" or "Did Not Survive" (2 options). In Part 2 (Iris), the target is "Setosa", "Versicolor", or "Virginica" (3 options). How does the Classification Report handle this difference?

<details>
<summary><strong>View Answer</strong></summary>

The Titanic dataset was a **Binary Classification** problem. The Iris dataset is a **Multi-class Classification** problem.

If you look at the Classification Report output in Step 2.6, you will notice that Scikit-Learn automatically adapts. Instead of calculating Precision and Recall for just two outcomes, it calculates individual Precision, Recall, and F1-Scores for *every single class* (all three flower types separately). It then provides an overall "macro average" and "weighted average" at the bottom to summarize the model's total performance across all categories.
</details>

<br>

**Question 3: Dataset Complexity**
Compare the accuracy scores you received on the Titanic model versus the Iris model. Why does the Iris model perform so much better?

<details>
<summary><strong>View Answer</strong></summary>

The Iris model typically achieves near-perfect accuracy (often 96% to 100%), whereas the Titanic model usually hovers around 80%. 

This happens for two reasons:
1. **Cleanliness:** The Iris dataset is a carefully measured, perfectly clean scientific dataset with no missing values.
2. **Predictability:** Petal and sepal sizes have a very strict, mathematical correlation to a flower's biological species. In contrast, human survival on the Titanic involved a massive amount of luck and chaos, making it much harder for a mathematical algorithm to predict perfectly.
</details>

---

### Part 3: Guided Regression (Auto MPG Dataset)

**Goal:** Learn the regression pipeline by predicting a car's fuel efficiency (Miles Per Gallon, or MPG) using a Linear Regression algorithm. 

Because we are predicting a continuous number (MPG) rather than a category, this is a **Regression** task.

#### Step 3.1: Load Pre-Cleaned Auto MPG Data
This dataset contains attributes of various cars (like weight, horsepower, and cylinders). The target variable we want to predict is `mpg`.

```python
# Load pre-cleaned Auto MPG dataset
url_mpg = "https://raw.githubusercontent.com/ML-Course-2026/session3/refs/heads/main/datasets/cars/auto-mpg.csv"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']

# The na_values="?" parameter tells Pandas to treat "?" as a missing value (NaN)
mpg_df = pd.read_csv(url_mpg, names=column_names, na_values="?", header=0)

# Basic Cleaning (We will dive much deeper into this in Part 5)
# 1. Drop the car name because it is a text identifier, not a mathematical feature.
mpg_df = mpg_df.drop('car_name', axis=1) 

# 2. Fill missing horsepower values with the median.
mpg_df['horsepower'].fillna(mpg_df['horsepower'].median(), inplace=True)

# 3. Drop 'origin' for now. It is a category (1=USA, 2=Europe, 3=Japan), but 
# treating it as a raw number confuses the model. We will fix this properly in Part 5.
mpg_df = mpg_df.drop('origin', axis=1)

# Drop any remaining rows with missing data
mpg_df.dropna(inplace=True) 

print("Auto MPG Data (first 5 rows):")
print(mpg_df.head())
print("\nData Info:")
mpg_df.info()
```

#### Step 3.2: Define Features (X) and Target (y)
We separate the target we want to predict (`mpg`) from the features we will use to make the prediction.

```python
# Define features (X) and target (y)
X_mpg = mpg_df.drop('mpg', axis=1)
y_mpg = mpg_df['mpg']

print("\nFeatures (X):")
print(X_mpg.head(2))
print("\nTarget (y):")
print(y_mpg.head(2))
```

#### Step 3.3: Split Data
We split the data into 80% training and 20% testing. 

```python
# Split the data
X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg = train_test_split(
    X_mpg, y_mpg, test_size=0.2, random_state=42
)

print(f"\nTraining set shape: X={X_train_mpg.shape}, y={y_train_mpg.shape}")
print(f"Testing set shape: X={X_test_mpg.shape}, y={y_test_mpg.shape}")
```

> [!NOTE]  
> Notice that we did **not** use `stratify=y_mpg` here like we did in Parts 1 and 2. Stratification ensures an equal balance of categories. Because regression deals with infinite possible continuous numbers (e.g., 25.1 MPG, 25.2 MPG), stratification does not mathematically apply here.

#### Step 3.4: Create and Train a Linear Regression Model
Unlike a Decision Tree that creates a flowchart of "if/then" rules, a Linear Regression model uses math to draw a "line of best fit" through the data points.

```python
# Create a Linear Regression model
lr_mpg = LinearRegression()

# Train the model on the training data
lr_mpg.fit(X_train_mpg, y_train_mpg)

print("\nLinear Regression model trained successfully.")
```

#### Step 3.5: Make Predictions on Test Data
We pass the hidden test features (`X_test_mpg`) to our trained model and ask it to guess the MPG for those cars.

```python
# Make predictions
y_pred_mpg = lr_mpg.predict(X_test_mpg)

# Display first 5 predictions vs actual values
print("\nFirst 5 Predictions:", y_pred_mpg[:5])
print("First 5 Actual Values:", y_test_mpg[:5].values)
```

#### Step 3.6: Evaluate the Model
Because we are predicting a continuous number, concepts like "Accuracy" (being 100% exactly right) do not apply. If a car's actual MPG is 25.0, and our model predicts 24.9, it is technically "wrong," but it is still a very good prediction. 

Therefore, for regression, we evaluate the model by measuring the **Error** (the distance between the prediction and the actual value).

```python
# Evaluate the model
mae_mpg = mean_absolute_error(y_test_mpg, y_pred_mpg)
mse_mpg = mean_squared_error(y_test_mpg, y_pred_mpg)
rmse_mpg = np.sqrt(mse_mpg) # We calculate RMSE by taking the square root of MSE
r2_mpg = r2_score(y_test_mpg, y_pred_mpg)

print(f"\nMean Absolute Error (MAE): {mae_mpg:.4f}")
print(f"Mean Squared Error (MSE): {mse_mpg:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_mpg:.4f}")
print(f"R-squared (R²): {r2_mpg:.4f}")
```

> [!IMPORTANT]  
> **Understanding Regression Metrics:**
> *   **MAE:** The average of all the errors. If MAE is 2.5, it means our model's predictions are off by an average of 2.5 MPG. It is the easiest to understand.
> *   **MSE:** Averages the *squares* of the errors. Squaring the numbers heavily penalizes the model for making massive mistakes, but it changes the units (e.g., "MPG squared").
> *   **RMSE:** The square root of MSE. This brings the error back into the original units (MPG), making it readable while still penalizing large errors.
> *   **R² (R-Squared):** A score from 0 to 1 indicating how well the model fits the data. 1.0 is a perfect fit. 0.0 means the model is no better than just guessing the average MPG every single time.

#### Step 3.7: Visualize Predictions vs Actuals
Numbers are great, but plotting the data helps us truly understand how the model behaves. 

```python
# Plot Actual vs Predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_mpg, y_pred_mpg, alpha=0.7)

# Draw a red dashed line representing "Perfect Prediction"
plt.plot([y_test_mpg.min(), y_test_mpg.max()], [y_test_mpg.min(), y_test_mpg.max()], '--r', linewidth=2) 
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs. Predicted MPG')
plt.show()

# Plot Residuals (The Errors)
residuals_mpg = y_test_mpg - y_pred_mpg
plt.figure(figsize=(10, 6))
sns.histplot(residuals_mpg, kde=True)
plt.xlabel('Residuals (Actual MPG - Predicted MPG)')
plt.title('Distribution of Residuals')
plt.show()
```

> [!TIP]  
> **How to read these charts:**
> 1.  **Scatter Plot:** You want all the blue dots to hug the red dashed line as tightly as possible. Dots far away from the line are poor predictions.
> 2.  **Residual Plot:** This shows the distribution of your model's mistakes. Ideally, this should look like a bell curve centered perfectly on `0`. If the peak is shifted to the left or right, your model is systematically predicting too high or too low.

###  Knowledge Check (Part 3)

Review the concepts covered in this section. Try to answer the questions before viewing the solutions.

**Question 1: The Accuracy Trap**
If you try to run the code `accuracy_score(y_test_mpg, y_pred_mpg)` on this model, Python will throw a massive error and crash. Why?

<details>
<summary><strong>View Answer</strong></summary>

`accuracy_score` is strictly a **Classification** metric. It checks if the prediction perfectly matches the target (True or False). 

In Regression, predictions are almost never perfectly exact. If the actual MPG is 20.0 and the model predicts 20.000001, `accuracy_score` will mark it as completely wrong, which is useless. For continuous numbers, we must measure the *distance* of the error (MAE/RMSE), not boolean correctness.
</details>

<br>

**Question 2: MAE vs RMSE**
Imagine you have a model predicting housing prices. The MAE is `$10,000`, but the RMSE is `$50,000`. Why would the RMSE be so much higher than the MAE, and what does this tell you about the model's performance?

<details>
<summary><strong>View Answer</strong></summary>

RMSE penalizes *large* errors significantly more than MAE because it squares the errors before averaging them. 

If RMSE is much higher than MAE, it tells you that while the model is usually quite accurate (off by about $10,000 on average), there are a handful of predictions where the model makes **massive, catastrophic mistakes** (e.g., overvaluing a house by $200,000). The squaring effect of RMSE highlights these outliers.
</details>

<br>

**Question 3: Missing Values**
In Step 3.1, we filled the missing `horsepower` values with the median horsepower of the entire dataset. Why did we use the median instead of simply filling the missing spaces with `0`?

<details>
<summary><strong>View Answer</strong></summary>

If you fill a missing horsepower value with `0`, you are telling the machine learning algorithm that the car literally has zero horsepower. The algorithm will treat that `0` as a real, mathematical fact and adjust its line of best fit accordingly, which will ruin the prediction (a car with 0 horsepower would not move). 

Filling missing values with the median (or mean) is a safe way to preserve the row of data without wildly distorting the mathematical average of the dataset.
</details>


<br>


**Question 4: Why Is Stratification Not Used in Regression?**
Why did we **not** use `stratify=y_mpg` in the regression example (e.g., predicting MPG) like we did in the classification examples (e.g., Iris, Titanic datasets)? Explain why stratification doesn't apply in the context of regression.

<details>  
<summary><strong>View Answer</strong></summary>

In classification problems, stratification is often used to ensure that the data is split evenly across the different classes, making sure each split (training and testing) has the same proportion of categories. This is crucial when working with imbalanced datasets to avoid biased models that overfit or underfit due to uneven class distributions.

However, **stratification does not apply in regression** problems, such as predicting continuous values like **MPG** (miles per gallon). In regression, the target variable is continuous, meaning it can take on any value within a range (e.g., 25.1 MPG, 25.2 MPG, etc.). This makes it impossible to categorize the target variable into discrete "classes" for stratification.

**Example:**
In the Iris dataset (classification), you have classes like 'setosa', 'versicolor', and 'virginica'. If you stratify by the target variable, the data will be split proportionally to ensure each category is equally represented. In the Titanic dataset (classification), you also have categories like 'survived' or 'did not survive'.

In contrast, with a regression problem like predicting **MPG**, where values like 25.1, 25.2, and 25.3 are all possible and continuous, there's no meaningful way to "split" the data into strata. Stratifying by the target would not help because the distribution of values is not discrete.

Thus, in regression tasks, it's more common to rely on **random splitting** of the dataset into training and test sets, without stratification, to ensure that all possible values of the target variable are present in both splits but without forcing a specific balance between categories.

**In summary:**

* **Stratification** works for **classification** (discrete categories).
* **Regression** deals with continuous values, so **stratification is not necessary** or meaningful.

For example, in the regression task predicting **MPG**, we would use a simple split without stratification:

```python
train_test_split(X, y_mpg, test_size=0.2, random_state=42)
```

This ensures a proper random distribution of the data without trying to force balance over continuous values.

</details>

<br>


---

### Part 4: Practice Regression (Diabetes Dataset)

> [!NOTE]  
> Just like in Part 2, the original plan for this lab was for you to write this code from scratch. We have provided the sample solution below. Your task is to run the code and study how the regression pipeline is applied to a completely different set of data.

**Goal:** Apply the regression pipeline to the `scikit-learn` Diabetes dataset. This dataset contains physiological measurements of patients, and the goal is to predict how much their disease will progress one year later.

#### Step 4.1: Load Diabetes Data
This dataset is built into `sklearn`. The features (like BMI, blood pressure, etc.) have already been scaled and cleaned by the dataset creators.

```python
# Load the Diabetes dataset
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Add the target column to the dataframe
diabetes_df['target'] = diabetes.target 

print("Diabetes Data (first 5 rows):")
print(diabetes_df.head())
print("\nData Info:")
diabetes_df.info()
```

> [!TIP]  
> **What exactly is the "Target" here?** 
> In the Auto MPG dataset, the target was obvious (Miles Per Gallon). In this dataset, the target is a numerical score (ranging from roughly 25 to 346) that represents how much the patient's diabetes progressed exactly one year after the baseline measurements were taken. A higher number means worse disease progression.

#### Step 4.2: Define Features (X) and Target (y)
We separate the target score from the patient features.

```python
# Define X_diabetes and y_diabetes
X_diabetes = diabetes_df.drop('target', axis=1)
y_diabetes = diabetes_df['target']

# Print shapes to verify
print(f"\nDiabetes features shape: {X_diabetes.shape}")
print(f"Diabetes target shape: {y_diabetes.shape}")
```

#### Step 4.3: Split Data
We split into 80% training and 20% testing. Again, because this is regression, we do not use `stratify`.

```python
# Split the data into X_train, X_test, y_train, y_test
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

# Print shapes to verify
print(f"\nTraining set shape: X={X_train_diabetes.shape}, y={y_train_diabetes.shape}")
print(f"Testing set shape: X={X_test_diabetes.shape}, y={y_test_diabetes.shape}")
```

#### Step 4.4: Create and Train a Linear Regression Model
We initialize a new Linear Regression model and fit it strictly to the training data.

```python
# Create and train the model 
lr_diabetes = LinearRegression()
lr_diabetes.fit(X_train_diabetes, y_train_diabetes)

print("\nDiabetes Linear Regression model trained.")
```

#### Step 4.5: Make Predictions
We pass the hidden test data to the model and ask it to predict the disease progression scores.

```python
# Make predictions 
y_pred_diabetes = lr_diabetes.predict(X_test_diabetes)

# Display first 5 predictions vs actual values
print("\nFirst 5 Predictions:", y_pred_diabetes[:5])
print("First 5 Actual Values:", y_test_diabetes[:5].values)
```

#### Step 4.6: Evaluate the Model
We calculate the error metrics to see how close our predictions are to the actual patient outcomes.

```python
# Evaluate the model 
mae_diabetes = mean_absolute_error(y_test_diabetes, y_pred_diabetes)
mse_diabetes = mean_squared_error(y_test_diabetes, y_pred_diabetes)
rmse_diabetes = np.sqrt(mse_diabetes)
r2_diabetes = r2_score(y_test_diabetes, y_pred_diabetes)

print(f"\nDiabetes Model Evaluation:")
print(f"MAE: {mae_diabetes:.4f}")
print(f"MSE: {mse_diabetes:.4f}")
print(f"RMSE: {rmse_diabetes:.4f}")
print(f"R²: {r2_diabetes:.4f}")
```

#### Step 4.7: Visualize Predictions vs Actuals
Visualizing the results helps us identify if our model has any specific biases (e.g., if it struggles to predict very high disease progression).

```python
# Plot Actual vs Predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_diabetes, y_pred_diabetes, alpha=0.7)
plt.plot([y_test_diabetes.min(), y_test_diabetes.max()], [y_test_diabetes.min(), y_test_diabetes.max()], '--r', linewidth=2)
plt.xlabel('Actual Progression')
plt.ylabel('Predicted Progression')
plt.title('Actual vs. Predicted Diabetes Progression')
plt.show()

# Plot Residuals
residuals_diabetes = y_test_diabetes - y_pred_diabetes
plt.figure(figsize=(10, 6))
sns.histplot(residuals_diabetes, kde=True)
plt.xlabel('Residuals (Actual - Predicted)')
plt.title('Distribution of Residuals (Diabetes)')
plt.show()
```

### Knowledge Check (Part 4)

Review the concepts covered in this section. Try to answer the questions before viewing the solutions.

**Question 1: Interpreting the R-Squared (R²) Score**
Compare the R² score you received for the Auto MPG model (Part 3) with the R² score for the Diabetes model (Part 4). 
1. Which model performed better? 
2. What does a lower R² score tell you about the dataset itself?

<details>
<summary><strong>View Answer</strong></summary>

1. The Auto MPG model performed significantly better. The Auto MPG R² score usually hovers around `0.80` (80% of the variance explained), while the Diabetes R² score usually hovers around `0.45` (only 45% of the variance explained).
2. A lower R² score tells you that the features provided to the model (BMI, blood pressure, etc.) are **not sufficient** to perfectly predict disease progression. Human biology is incredibly complex. A score of ~0.45 means our model captures some real underlying trends, but there is a massive amount of "noise" or unmeasured factors (like genetics or daily diet) affecting the patient that our model simply cannot see.
</details>

<br>

**Question 2: Making Sense of the MAE**
In Step 4.6, you printed the Mean Absolute Error (MAE). If the MAE is roughly `42.7`, how do you explain that number to a doctor who has no background in data science?

<details>
<summary><strong>View Answer</strong></summary>

You would explain it by saying: *"When this AI predicts a patient's disease progression score for next year, its guess will be off by an average of 43 points, either too high or too low."*

Because the disease progression scores range from roughly 25 to 346, being off by 43 points means the model provides a decent "ballpark" estimate, but it is not precise enough to make highly exact medical decisions.
</details>

<br>

**Question 3: Examining the Scatter Plot**
Look closely at the first chart generated in Step 4.7 (Actual vs. Predicted). Notice that as the Actual Progression (the x-axis) gets higher, the blue dots tend to fall *below* the red dashed line. What does this indicate about the model's behavior?

<details>
<summary><strong>View Answer</strong></summary>

When dots fall below the red line of perfect prediction, it means the model is **underpredicting**. 

This chart visually proves that our Linear Regression model struggles with the sickest patients. When a patient's actual disease progression is very severe (e.g., above 250), our model consistently guesses numbers that are too low. In a medical context, this is a critical flaw, because underpredicting disease severity could lead to a patient not receiving enough care.
</details>



---

### Part 5: The Reality - Data Preprocessing

**Goal:** Understand *why* preprocessing is crucial and see *how* the raw Auto MPG data needed cleaning before we could use it effectively in Part 3.

**The Scenario:** In Parts 1-4, we used data that was mostly ready for modeling. Real-world data is almost never that clean! It often has missing values, non-numeric data types, and features on different scales. Models usually require numerical input and perform better when data is scaled.

**Let's revisit the Auto MPG dataset, but load the *raw* version this time.**

**Step 5.1: Load RAW Data and Initial Inspection**
Notice the `na_values='?'`: this tells pandas to treat '?' as missing.

```python
import pandas as pd

# Load the raw data again
url_mpg = "https://raw.githubusercontent.com/ML-Course-2026/session3/main/datasets/cars/auto-mpg.csv"

# Define column names (to ensure consistency)
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']

# Read CSV correctly (comma-separated)
raw_mpg_df = pd.read_csv(url_mpg, names=column_names, na_values="?", header=0)

# Display first rows
print("Raw Auto MPG Data (first 5 rows):")
print(raw_mpg_df.head())

# Data info
print("\nRaw Data Info:")
raw_mpg_df.info()

# Check for missing values
print("\nMissing values count:")
print(raw_mpg_df.isnull().sum())
```
*   **Observation:** `horsepower` has missing values (6). Also notice `horsepower` is an `object` (text) type, because of the '?' before we handled `na_values`. `origin` is numeric but represents categories (1: USA, 2: Europe, 3: Japan). `car_name` is text and probably not useful as a raw feature.

**Step 5.2: Handling Missing Values (Imputation)**
We need to fill or remove missing values. For `horsepower`, let's fill with the median.

```python
# Calculate median horsepower (ignoring NaNs)
median_hp = raw_mpg_df['horsepower'].median()
print(f"\nMedian horsepower: {median_hp}")

# Fill missing horsepower values
raw_mpg_df['horsepower'].fillna(median_hp, inplace=True)

# Verify missing values are handled
print("\nMissing values count after imputation:")
print(raw_mpg_df.isnull().sum())
```

**Step 5.3: Handling Categorical Features (Encoding)**
`origin` is categorical. Treating it as a number (1, 2, 3) implies an order and distance that doesn't exist. We should use One-Hot Encoding. `car_name` is too unique; we'll drop it.

```python
# Drop the car name column
raw_mpg_df = raw_mpg_df.drop('car_name', axis=1)

# Use One-Hot Encoding for 'origin'
# This creates new columns like 'origin_1', 'origin_2', 'origin_3'
raw_mpg_df = pd.get_dummies(raw_mpg_df, columns=['origin'], prefix='origin', drop_first=False) # drop_first=False keeps all origins explicit

print("\nData after One-Hot Encoding 'origin' (first 5 rows):")
print(raw_mpg_df.head())
print("\nData Info after Encoding:")
raw_mpg_df.info()
```
*   **Observation:** `origin` is gone, replaced by `origin_1`, `origin_2`, `origin_3`. All columns are now numeric.

**Step 5.4: Feature Scaling (Standardization)**
Features like `weight` (thousands) and `acceleration` (tens) have vastly different scales. Many models (including Linear Regression, though it's less sensitive) benefit from scaling features to have zero mean and unit variance (Standardization).

```python
# Separate target from features *before* scaling
y_processed = raw_mpg_df['mpg']
X_processed = raw_mpg_df.drop('mpg', axis=1)

# Identify numerical columns to scale (exclude the one-hot encoded origin columns for this example, though scaling them doesn't hurt)
numerical_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']

# Create the scaler
scaler = StandardScaler()

# Fit and transform the numerical columns
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

print("\nFeatures after Standardization (first 5 rows):")
print(X_processed.head())

print("\nDescription of Scaled Features:")
print(X_processed[numerical_cols].describe()) # Mean should be close to 0, std dev close to 1
```

**Step 5.5: Data is Ready!**
*Now* the `X_processed` and `y_processed` dataframes are truly ready for the Split -> Train -> Predict -> Evaluate steps we performed in Part 3. The preprocessing steps (handling missing values, encoding categoricals, scaling) are essential for most real-world machine learning tasks.

**Key Takeaways from Part 5:**
*   Real-world data requires cleaning and preparation (preprocessing).
*   Common steps include handling missing data (imputation), converting categorical features to numbers (encoding), and scaling numerical features.
*   Tools like `pandas` for data manipulation and `scikit-learn`'s `StandardScaler` and `OneHotEncoder` (often used within `ColumnTransformer` and `Pipeline` for more complex workflows) are essential.
*   Preprocessing makes data suitable for modeling and often improves model performance.

---

## Recap

You have:
- Explored the fundamentals of supervised learning (classification and regression).
- Implemented the standard ML pipeline using `scikit-learn`.
- Trained and evaluated Decision Tree and Linear Regression models.
- Visualized model results.
- Gained an appreciation for the critical role of data preprocessing in real-world scenarios.

This lab provides a foundation. There are many more algorithms, evaluation techniques, and preprocessing methods to explore in the world of machine learning!



<details>

<summary>Take home message</summary>

**1. What is Supervised Learning?**

*   **Definition:** A type of Machine Learning where models learn from **labeled data**. Each data point has input features and a known correct output (label/target).
*   **Goal:** To learn a function that maps inputs to outputs, enabling predictions on new, unseen data.
*   **Two Main Types:**
    *   **Classification:** Predicts a discrete **category** (e.g., Survive/Not Survive, Spam/Not Spam, Iris Species).
    *   **Regression:** Predicts a continuous **numerical value** (e.g., Car MPG, Diabetes Progression Score, House Price).

**2. The Standard Machine Learning Pipeline**

A consistent workflow applies to most supervised learning tasks:

1.  **Load Data:** Import data (e.g., using `pandas`).
2.  **Explore & Preprocess Data:** Understand, clean, and transform data (Crucial!).
3.  **Split Data:** Divide into **Training Set** (for learning) and **Testing Set** (for evaluation) using `train_test_split`.
4.  **Choose & Create Model:** Select an algorithm (e.g., `DecisionTreeClassifier`, `LinearRegression`).
5.  **Train Model:** Fit the model to the **training data** (`model.fit(X_train, y_train)`).
6.  **Make Predictions:** Use the trained model on the **test data** (`model.predict(X_test)`).
7.  **Evaluate Model:** Assess performance using appropriate metrics by comparing predictions to actual test labels.

**3. Classification Concepts (Parts 1 & 2)**

*   **Goal:** Assign data points to predefined categories.
*   **Example Model:** Decision Tree (`sklearn.tree.DecisionTreeClassifier`).
    *   Builds a tree-like structure to make decisions based on feature values.
    *   `max_depth` parameter helps control complexity and prevent overfitting.
*   **Evaluation Metrics:**
    *   **Accuracy:** Overall percentage of correct predictions.
    *   **Precision:** Accuracy of positive predictions (TP / (TP + FP)).
    *   **Recall (Sensitivity):** Ability to find all actual positive instances (TP / (TP + FN)).
    *   **F1-Score:** Harmonic mean of Precision and Recall, useful for balancing them.
    *   `sklearn.metrics.accuracy_score`, `sklearn.metrics.classification_report`.

**4. Regression Concepts (Parts 3 & 4)**

*   **Goal:** Predict a numerical value.
*   **Example Model:** Linear Regression (`sklearn.linear_model.LinearRegression`).
    *   Finds the best linear relationship between features and the target.
*   **Evaluation Metrics:**
    *   **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values. Easy to interpret.
    *   **Mean Squared Error (MSE):** Average squared difference. Penalizes larger errors more.
    *   **Root Mean Squared Error (RMSE):** Square root of MSE. In the same units as the target.
    *   **R-squared (R²):** Proportion of target variance explained by the model (0 to 1, higher is better).
    *   `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.mean_squared_error`, `sklearn.metrics.r2_score`.
*   **Visualization:**
    *   **Actual vs. Predicted Plot:** Scatter plot to visually check prediction quality (points near diagonal line are good).
    *   **Residual Plot:** Histogram or scatter plot of errors (Actual - Predicted). Ideally centered around zero with no clear pattern.

**5. Data Preprocessing (Part 5)**

*   **Why it's Crucial:** Real-world data is often "messy". Models require specific data formats and often perform better with processed data.
*   **Common Steps:**
    *   **Handling Missing Values:**
        *   Identify using `.isnull().sum()`.
        *   Strategies: Remove rows/columns, or **Impute** (fill) with mean, median (for numerical), or mode (for categorical).
    *   **Encoding Categorical Features:**
        *   Convert text/categorical data into numbers.
        *   **One-Hot Encoding** (`pd.get_dummies` or `sklearn.preprocessing.OneHotEncoder`) is common, creating binary columns for each category.
    *   **Feature Scaling:**
        *   Bring numerical features to a similar scale. Important for distance-based algorithms and gradient descent.
        *   **Standardization** (`sklearn.preprocessing.StandardScaler`): Transforms data to have zero mean and unit variance.
        *   **Normalization:** Scales data to a specific range (e.g., 0 to 1).

**6. Key Tools Used**

*   **Pandas:** For data loading, manipulation, and exploration.
*   **Scikit-learn:** The core library for ML models, splitting, preprocessing, and evaluation.
*   **Matplotlib & Seaborn:** For data visualization.

**Conclusion:** This lab provided hands-on experience with the fundamental workflow of supervised machine learning, covering both classification and regression tasks, model evaluation, and the essential step of data preprocessing using standard Python libraries.
</details>