import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.title("Adult Income Prediction – ML Models")

uploaded_file = st.file_uploader("Upload Adult Dataset CSV", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Strip spaces from string columns (VERY IMPORTANT)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Replace ? with NaN and drop
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        if col != "income":
            df[col] = LabelEncoder().fit_transform(df[col])

    # Encode target column
    df["income"] = LabelEncoder().fit_transform(df["income"])

    # Split X and y
    X = df.drop("income", axis=1)
    y = df["income"]

    # Model paths
    model_paths = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl"
    }

    # Load trained model (DO NOT FIT AGAIN)
    model = joblib.load(model_paths[model_name])

    # Predict
    y_pred = model.predict(X)

    # Output results
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))
