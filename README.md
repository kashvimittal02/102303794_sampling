# Credit Card Fraud Detection Using Sampling Techniques

## Objective

The objective of this project is to study how sampling techniques help handle imbalanced datasets and how different sampling strategies affect the performance of machine learning models. A highly imbalanced credit card dataset is used to evaluate and compare multiple sampling techniques across several ML models.

---

## Methodology

The project follows a structured pipeline:

### Step 1: Data Preprocessing

* The dataset is loaded using Pandas.
* Features and target variables are separated.
* Feature scaling is applied using StandardScaler to normalize the data.
* The class imbalance is analyzed.

---

### Step 2: Sampling Techniques

Five sampling strategies are applied to balance the dataset:

1. **Random Oversampling**

   * Duplicates minority class samples
   * Increases fraud cases to match the majority

2. **Random Undersampling**

   * Removes samples from the majority class
   * Reduces dataset size but balances classes

3. **SMOTE (Synthetic Minority Oversampling Technique)**

   * Generates synthetic minority samples
   * Improves generalization

4. **SMOTEENN**

   * Combines SMOTE with Edited Nearest Neighbor cleaning
   * Removes noisy samples

5. **SMOTETomek**

   * Combines SMOTE with Tomek link removal
   * Improves class separation

Each sampling technique creates a balanced dataset.

---

### Step 3: Machine Learning Models

Five ML models are trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

Each model is trained on every sampled dataset.

---

### Step 4: Model Training & Evaluation

* Each sampled dataset is split into training and testing sets (70:30)
* Models are trained using the training data
* Predictions are made on the test set
* Accuracy is calculated for performance comparison

---

## Result Table

The result table shows accuracy (%) of each model with each sampling method.
<img width="1227" height="274" alt="image" src="https://github.com/user-attachments/assets/4d05c700-b468-4b31-a40a-caf88088c003" />

### Interpretation

* Each row represents a machine learning model
* Each column represents a sampling technique
* Higher accuracy indicates better performance
* The best sampling technique for each model is selected based on maximum accuracy

---

## Result Graph

A bar chart visualization compares model performance across sampling techniques.
<img width="1247" height="768" alt="image" src="https://github.com/user-attachments/assets/927a53be-6b28-4e6a-9a0d-a5526bad3bf1" />

### Graph Insights

* Helps visually compare performance differences
* Highlights which sampling method works best
* Shows consistency or variability across models
* Makes interpretation easier than raw tables

---

## Key Observations

* Sampling significantly improves model performance
* Advanced techniques like SMOTE and hybrid methods often outperform simple sampling
* Different models respond differently to sampling strategies
* No single method is universally best

---

## Conclusion

This project demonstrates that:

* Handling class imbalance is critical for reliable ML performance
* Sampling techniques directly impact model accuracy
* Hybrid sampling methods often produce better results
* Choosing the right sampling technique depends on the model

Proper sampling leads to more accurate and fair fraud detection systems.

---

## How to Run the Project

1. Open Google Colab
2. Upload the dataset CSV file
3. Run the notebook cells sequentially
4. View the accuracy table and graph

---

## Tools & Libraries Used

* Python
* Pandas & NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib

---

Assignment project on sampling techniques and machine learning model comparison.
