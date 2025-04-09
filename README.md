# 📊 Bank Marketing KNN Classifier

A Streamlit web app to predict the outcome of a bank's marketing campaign using the K-Nearest Neighbors (KNN) algorithm. The prediction helps understand whether a client will subscribe to a term deposit (`yes` or `no`).

---

## 📁 Dataset

The dataset used is from the **Bank Marketing Data Set** available at the UCI Machine Learning Repository and Kaggle.

- [Download from Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)

---

## 🧠 ML Model

- **Model Used**: K-Nearest Neighbors Classifier (`KNeighborsClassifier`)
- **Preprocessing**:
  - Label Encoding for categorical columns
  - User selects value of **K**
  - Dynamic input from the sidebar
- **Evaluation**:
  - Accuracy plot of train vs test set for different K values
  - Visualization of target variable distribution

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Apekshaj04/BankMarketing.git
   cd BankMarketing
