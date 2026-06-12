# Credit Card Fraud Detection

A machine learning web app to detect fraudulent credit card transactions. Built with two models — Random Forest (sklearn) and a custom Neural Network (PyTorch) — with a live Streamlit interface.

## Live Demo
[Credit Card Fraud Detection App](https://credit-card-fraud-detection-using-ml-dyhneswqndchxzvxlydsxb.streamlit.app/)

## Screenshots

### App Interface
![App Interface](screenshots/image3.png)

### Input Features
![Input Features](screenshots/image1.png)

### Prediction Result
![Prediction Result](screenshots/image2.png)

## About
Credit card fraud detection is a classic imbalanced classification problem. This project implements and compares two approaches — a Random Forest classifier and a custom PyTorch neural network — both trained on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud).

## Model Performance

| Model | ROC-AUC | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Random Forest (sklearn) | 0.9999 | 0.89 | 1.00 | 1.00 |
| Neural Network (PyTorch) | 0.9998 | 1.00 | 1.00 | 1.00 |

## Tech Stack
- Python
- PyTorch (Neural Network)
- Scikit-learn (Random Forest)
- imbalanced-learn (SMOTE)
- Streamlit (Web App)
- Pandas, NumPy, Matplotlib, Seaborn

## Project Structure
```
FRAUD-DETECTION/
├── data/
│   └── creditcard.csv
├── screenshots/
├── app.py              # Streamlit web app
├── train.py            # Random Forest training
├── train_pytorch.py    # PyTorch Neural Network training
├── fraud_model.pkl     # Saved Random Forest model
├── requirements.txt
└── README.md
```

## How to Run Locally

**1. Clone the repo:**
```bash
git clone https://github.com/ROHIT-25607/credit-card-fraud-detection-using-ML.git
cd credit-card-fraud-detection-using-ML
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Download dataset:**
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

**4. Train Random Forest model:**
```bash
python train.py
```

**5. Train PyTorch Neural Network:**
```bash
python train_pytorch.py
```

**6. Run the app:**
```bash
streamlit run app.py
```

## How it Works
1. Dataset loaded and preprocessed (StandardScaler on Time and Amount)
2. SMOTE applied on training data only to handle class imbalance
3. Two models trained — Random Forest and a 4-layer PyTorch Neural Network (BatchNorm + Dropout)
4. Models evaluated on original imbalanced test set using AUC, F1, Precision, Recall
5. Streamlit app loads model and makes real-time predictions
