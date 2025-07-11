# Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions, based on the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Dataset

- 284,807 transactions
- Features: V1 to V28 (PCA components), Time, Amount
- Target: 1 = Fraud, 0 = Legit

## Methods

- Data scaling
- SMOTE for class imbalance
- Random Forest classifier
- Evaluation: ROC-AUC, F1-score, Confusion Matrix

## Results

- ROC-AUC: 0.92
- F1-score: 0.89

## How to Run

1. Download `creditcard.csv` and place it in the `data/` folder
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
