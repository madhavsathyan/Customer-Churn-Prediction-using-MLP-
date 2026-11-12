# Customer Churn Prediction using a Multilayer Perceptron (MLP)

An end-to-end, academic deep learning project on the **IBM Telco Customer Churn** dataset (`blastchar/telco-customer-churn`).

---

## 📌 Project Overview
Customer churn—the rate at which customers discontinue subscriptions—poses a critical financial challenge in the telecommunications sector. Because acquiring a new customer is 5 to 7 times more expensive than retaining an existing one, proactive churn prediction enables telecommunications operators to design targeted retention campaigns, protect annual recurring revenue (ARR), and maximize Customer Lifetime Value (CLV).

This project implements a fully regularized **Multilayer Perceptron (MLP)** in TensorFlow/Keras to predict churn probability, benchmarked against a baseline **Logistic Regression** model. The implementation adheres to rigorous academic and machine learning engineering standards:
- Strict prevention of data leakage (transformers fitted exclusively on the training split).
- Analytical NumPy backpropagation derived from scratch and verified against TensorFlow autodiff.
- Strategic handling of class imbalance using cost-sensitive class weights.
- Multi-metric evaluation and decision threshold tuning aligned with telecom business economics.
- Systematic permutation feature importance to uncover key churn drivers.

---

## 🗂️ Project Structure

```text
ai ml 2/
├── churn_mlp.ipynb                          # Complete, self-contained, executed Jupyter Notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # IBM Telco Customer Churn dataset
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation & technical overview
└── figures/                                 # High-resolution generated visual assets (300 DPI)
    ├── 01_class_distribution.png            # Target imbalance visualization
    ├── 02_outlier_boxplots.png              # Boxplots & IQR analysis for continuous features
    ├── 03_training_validation_loss.png      # Epoch-wise training & validation loss curves
    ├── 04_training_validation_accuracy.png  # Epoch-wise training & validation accuracy curves
    ├── 05_confusion_matrix.png              # Normalized and raw confusion matrix heatmaps
    ├── 06_roc_pr_curves.png                 # ROC curve (AUC) and Precision-Recall curve (AP)
    ├── 07_threshold_tuning.png              # Precision, Recall & F1 vs. Classification Threshold
    ├── 08_feature_importance.png            # Permutation feature importance ranking
    └── 09_model_comparison.png              # MLP vs. Logistic Regression metric comparison
```

---

## 🏛️ Neural Network Architecture

```text
Input Layer (30 Preprocessed Features)
         │
         ▼
Dense Layer (64 Units, ReLU Activation)     [Params: 30 * 64 + 64 = 1,984]
         │
         ▼
Dropout Layer (Rate = 0.3)                  [Params: 0]
         │
         ▼
Dense Layer (32 Units, ReLU Activation)     [Params: 64 * 32 + 32 = 2,080]
         │
         ▼
Dropout Layer (Rate = 0.3)                  [Params: 0]
         │
         ▼
Output Layer (1 Unit, Sigmoid Activation)   [Params: 32 * 1 + 1 = 33]
         │
         ▼
P(Churn | X) ∈ [0, 1]                       Total Trainable Parameters: 4,097
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
Ensure you have **Python 3.10+** (tested on Python 3.12).

### 2. Environment Setup
```bash
# Optional: create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Dataset Download
If `WA_Fn-UseC_-Telco-Customer-Churn.csv` is not present locally, the notebook includes an automated cell to fetch it via `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
```
Alternatively, download it manually from Kaggle ([blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` directly in the project root.

### 4. Running the Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook churn_mlp.ipynb
```
Select **Run All Cells** from the top menu. The notebook runs deterministically top-to-bottom.

---

## 📊 Summary of Experimental Results (Test Split)

| Metric | Baseline (Logistic Regression) | Multilayer Perceptron (MLP) | Absolute Difference | Business Impact / Operational Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 74.55% | 73.89% | -0.66% | Overall percentage of correct customer classifications |
| **Precision (Churn=1)** | 51.41% | 50.57% | -0.84% | Proportion of flagged retention alerts that were genuine churners |
| **Recall (Churn=1)** | 77.94% | **78.65%** | **+0.71%** | **Proportion of defecting subscribers successfully intercepted** |
| **F1-Score (Churn=1)** | 0.6195 | 0.6156 | -0.0039 | Harmonic mean balancing campaign spend and churn capture |
| **ROC-AUC** | 0.8446 | **0.8422** | -0.0024 | Global discriminative capability across all operating thresholds |
| **PR-AUC (Avg Prec)** | 0.6579 | 0.6443 | -0.0136 | Precision-Recall curve area under class imbalance |
| **Cost-Optimal Threshold** | - | **τ = 0.18 ($40.07/cust)** | **-$37.37** | **Yields 97.86% Recall, reducing business costs by ~48%** |

*(Exact values are generated deterministically in `churn_mlp.ipynb` with seed 42).*

---

## 🔍 Key Findings & Business Recommendations
1. **Contract Structure is Paramount**: Month-to-month contracts are by far the single strongest predictor of churn. Telcos should incentivize transitions to 1-year or 2-year agreements through initial bill credits or speed tier upgrades.
2. **First-Year Onboarding Danger Zone**: Churn is heavily concentrated in low-tenure customers (months 1–12). Dedicated onboarding proactive check-ins and customer satisfaction audits during this window can stem early defection.
3. **Fiber Optic Scrutiny**: Fiber optic subscribers display high churn rates despite high monthly spend. Investigation into service outages, technical support responsiveness, and competitor fiber pricing is strongly recommended.
4. **Electronic Check Friction**: Customers paying via electronic check churn at significantly higher rates than auto-debit or credit card users. Offering a monthly discount (\$5/mo) for auto-pay enrollment will directly improve customer retention.

---
Live demo link= https://madhavsathyan-customer-churn-prediction-using-mlp--app-efjv7s.streamlit.app/

## 👥 Viva & Oral Defense Highlights
The notebook concludes with a comprehensive academic Q&A covering:
- Mathematical comparison of **ReLU vs. Sigmoid/Tanh** activations.
- Optimization mechanics of the **Adam** algorithm (Momentum + RMSprop + Bias Correction).
- **Backpropagation** derivation via the multivariable chain rule.
- Methodological rationale for **StandardScaler** fitted strictly on training data.
- Stochastic regularization dynamics of **Dropout**.
- Domain-specific cost asymmetry explaining why **Recall** dominates **Accuracy** in churn contexts.
- **Class Weighting** vs. **SMOTE** for tabular imbalance.
- Inductive biases of **MLPs** vs. **Logistic Regression**.

