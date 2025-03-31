import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def get_clean_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    data = pd.read_csv(data_path)
    data = data.drop(columns=[col for col in ['id', 'Unnamed: 32'] if col in data.columns], errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def create_and_save_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:\n", classification_report(y_test, y_pred))
    
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "model.pkl"), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        data = get_clean_data()
        create_and_save_model(data)
    return pickle.load(open(model_path, "rb")), pickle.load(open(scaler_path, "rb"))


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    input_dict = {}
    for col in data.columns[:-1]:
        input_dict[col] = st.sidebar.slider(col, float(0), float(data[col].max()), float(data[col].mean()))
    return input_dict


def get_radar_chart(input_data):
    categories = list(input_data.keys())[:10]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=list(input_data.values())[:10], theta=categories, fill='toself', name='Features'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    return fig


def add_predictions(input_data, model, scaler):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is:")
    if prediction[0] == 0:
        st.success("Benign")
    else:
        st.error("Malignant")
    
    st.write("Probability of being benign:", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant:", model.predict_proba(input_array_scaled)[0][1])


def main():
    st.set_page_config(page_title="Breast Cancer Predictor", page_icon="👩‍⚕️", layout="wide")
    st.title("Breast Cancer Predictor")
    st.write("This app predicts whether a breast mass is benign or malignant based on cytology lab measurements.")
    
    model, scaler = load_model()
    input_data = add_sidebar()
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.plotly_chart(get_radar_chart(input_data))
    with col2:
        add_predictions(input_data, model, scaler)


if __name__ == '__main__':
    main()