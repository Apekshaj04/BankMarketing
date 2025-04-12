import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/Apekshaj04/BankMarketing/main/bank.csv", sep=";")

columns_to_drop = ['day', 'duration', 'pdays', 'default']
df.drop(columns=columns_to_drop, inplace=True)

columns_to_encode = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'poutcome']

label_encoders = {}
category_mappings = {}

for col in columns_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    category_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

Y = df['y']
X = df.drop(columns=['y'])

smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

st.title("📊 Bank Marketing KNN Classifier")

k = st.sidebar.slider("Select value of K (neighbors)", 1, 15, 5)

st.sidebar.header("📥 Enter Input Features")
user_input = []

for col in X.columns:
    if col in category_mappings:
        options = list(category_mappings[col].keys())
        selected = st.sidebar.selectbox(f"{col}", options)
        encoded = category_mappings[col][selected]
        user_input.append(encoded)
    else:
        default_val = int(X[col].mean())
        min_val = 0 if col in ['age', 'balance'] else int(X[col].min())
        value = st.sidebar.number_input(f"{col}", value=default_val, min_value=min_val)
        user_input.append(value)

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)
Y_pred_test = knn.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_pred_test)

if st.button("Predict"):
    input_array = np.array([user_input])
    prediction = knn.predict(input_array)
    st.success(f"✅ Prediction: **{prediction[0]}**")

st.metric("🎯 Test Accuracy", f"{test_accuracy*100:.2f}%")

st.subheader("📈 Target Variable Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x=Y, ax=ax1)
st.pyplot(fig1)

st.subheader("📉 KNN Accuracy vs K")
train_scores = []
test_scores = []
k_range = range(1, 21)

for i in k_range:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, Y_train)
    train_scores.append(model.score(X_train, Y_train))
    test_scores.append(model.score(X_test, Y_test))

fig2, ax2 = plt.subplots()
ax2.plot(k_range, train_scores, label='Train Accuracy', marker='o')
ax2.plot(k_range, test_scores, label='Test Accuracy', marker='x')
ax2.set_xlabel("K Value")
ax2.set_ylabel("Accuracy")
ax2.set_title("KNN Accuracy vs K")
ax2.legend()
st.pyplot(fig2)
