# Loan Default Prediction - Custom Logistic Regression with Batch Gradient Descent

## 📌 Objective

The goal is to build a binary classification model to predict whether a loan applicant will default or not using:
- Custom implementation of Logistic Regression using **Batch Gradient Descent** with **L2 regularization**
- Evaluation and comparison of performance metrics

---

## 📁 Dataset

- Input CSV: `applicants_preprocessed.csv`
- Target variable: `default` (binary: 0 - No Default, 1 - Default)
- Feature set includes numerical and categorical fields (converted to numeric)
- Additional text field: `applicant_statement` (will be used in Option A: BERT fine-tuning)

---

## ✅ Completed Implementation (Option A - Part 1)

### 1. Data Preparation
- CSV file is loaded using pandas
- Non-numeric values are coerced into NaN and handled properly
- Bias term is added to the features
- Data is split into 80% training and 20% testing sets

### 2. Model: Logistic Regression from Scratch
- **Sigmoid:** Numerically stable function
- **Loss:** Cross-entropy with L2 regularization
- **Gradient:** Computed over full batch with regularization (bias excluded)
- **Batch Gradient Descent:** Trains weights using specified learning rate and λ (L2 regularization strength)

### 3. Hyperparameters
- Learning Rates tested: `[0.01, 0.05, 0.1]`
- Regularization values (λ): `[0.1, 1]`
- Epochs: `1000`

### 4. Evaluation Metrics
Evaluated on test set:
- Accuracy
- Precision
- Recall
- ROC-AUC

### 5. Visualizations
- Line plot: **Loss vs Epochs** for different hyperparameter combinations

---

## 🔍 Observations & Inference

- Proper convergence for most learning rates and regularization values
- `lr = 0.05` was most stable across λ values
- Higher λ (like 1.0) prevented overfitting by shrinking weights
- `applicant_statement` column **not used yet**, but ready for **Option A (BERT-based fine-tuning)**

---

## 🧪 Requirements

Make sure the following Python libraries are installed:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## ▶️ How to Run

1. Ensure your working directory contains the following file:
    ```
    applicants_preprocessed.csv
    ```

2. Run the Python script or notebook (`applicants_defaulter.py` or `.ipynb`) using any Python 3.7+ environment:

```bash
python applicants_defaulter.py
```

Or launch a Jupyter notebook:

```bash
jupyter notebook applicants_defaulter.ipynb
```

3. Output includes:
    - Console logs of training loss
    - Line plot: Loss vs. Epochs
    - Test set metrics: Accuracy, Precision, Recall, ROC-AUC

---

## 🔜 Next Steps (Option A)

- Fine-tune BERT on the `applicant_statement` column to predict loan default
- Compare BERT model performance with the custom logistic regression implementation

---

👨‍💻 Author: Manoj Dakuwa 
📅 Date: May 2025  
