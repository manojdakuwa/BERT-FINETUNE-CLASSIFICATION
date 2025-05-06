
# Loan Default Prediction with Custom Logistic Regression and Fine-Tuned BERT

## 🔍 Project Overview
This project tackles a binary classification task — predicting loan default — using two approaches:
1. A custom logistic regression implemented from scratch using NumPy.
2. A fine-tuned BERT model (transformer-based) to extract meaningful features from the `applicant_statement` text field.

These embeddings are later combined with other structured features and passed to the logistic regression model for final prediction.

---

## 📁 Dataset Information

The dataset (`applicants.csv`) consists of:
- **Structured Fields**: Numeric features like income, loan amount, age, etc.
- **Text Field**: `applicant_statement` — a field describing applicant's financial situation (synthetically generated).

An updated dataset with generated statements is saved as:  
✅ `applicants_updated.csv`

---

## 💻 Environment & Hardware Requirements

### 1. Software Setup
- Python 3.8+
- Libraries:
  ```bash
  pip install numpy pandas scikit-learn transformers torch matplotlib
  ```

### 2. Hardware Requirements
- ✅ **CPU** is sufficient for traditional logistic regression (Option B).
- ⚠️ **GPU is strongly recommended** for BERT fine-tuning (Option A) due to its computational intensity.
  - Minimum: 8 GB VRAM (e.g., NVIDIA Tesla T4 or RTX 3060)
  - Setup: Google Colab, Kaggle Kernels, or local CUDA-enabled machine.

---

## 🚀 How to Run the Project

### Step 1: Update Dataset with Statements
```python
python applicants_defaulter.py
```
This will generate a column `applicant_statement` based on risk levels and save the dataset as `applicants_updated.csv`.

### Step 2: Option B - Custom Logistic Regression (Baseline)
```python
python logistic_regression_numpy.py
```
This trains a logistic regression model from scratch using batch gradient descent and L2 regularization.

### Step 3: Option A - Fine-Tune BERT & Integrate
```python
python fine_tune_bert.py
```
- Fine-tunes `bert-base-uncased` on `applicant_statement`
- Extracts classification logits as sentence embeddings
- Merges them with original structured features
- Applies logistic regression

---

## 📊 Evaluation Metrics

For both options, we evaluate using:
- Accuracy
- Precision
- Recall
- ROC-AUC

F1-score was explicitly **not required** by the prompt.

---

## 📈 Visualizations

### 🔹 Loss vs Epochs (Custom Logistic Regression)
_A graph showing loss convergence for different learning rates and regularization values._

![Loss Curve](loss_curve.png)

---

## 🧠 Key Learnings & When to Use What

| Approach | When to Use | Pros | Cons |
|---------|-------------|------|------|
| **Custom Logistic Regression** | Small, structured data | Simple, interpretable | Cannot model text directly |
| **Fine-Tuned BERT + Logistic Regression** | Rich text like `applicant_statement` | Leverages semantic meaning | Resource-intensive, needs GPU |

---

## 📂 Output Files

| File | Description |
|------|-------------|
| `applicants_updated.csv` | Dataset with added `applicant_statement` |
| `bert_loan_default_model/` | Fine-tuned BERT model directory |
| `X_final.npy`, `y_final.npy` | Combined features + labels for final logistic regression |
| `loss_curve.png` | Loss convergence plot |

---

## ✅ Notes
- Learning rates `[0.01, 0.05, 0.1]` and regularization values `[0.1, 1]` were selected based on experimentation and convergence behavior.
- The BERT classifier logits were used as embeddings rather than mean-pooled features for simplicity.
