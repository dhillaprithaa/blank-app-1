import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.write("### Debug: Script Dimulai")  # Debugging

# Load models
try:
    with open("models/kmeans_models.pkl", "rb") as f:
        kmeans = pickle.load(f)
    st.write("### Debug: KMeans Model Berhasil Dimuat")
except Exception as e:
    st.write(f"Error loading KMeans model: {e}")

try:
    with open("models/pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    st.write("### Debug: PCA Model Berhasil Dimuat")
except Exception as e:
    st.write(f"Error loading PCA model: {e}")

# Load dataset
try:
    df = pd.read_csv("data/Country-data.csv")
    st.write("### Debug: Dataset Berhasil Dimuat")
except Exception as e:
    st.write(f"Error loading dataset: {e}")

columns = df.drop(columns=["country"]).columns

# Preprocessing
scaler = StandardScaler()
df_clean = df.drop(columns=["country"], errors="ignore")
scaler.fit(df_clean)
st.write("### Debug: Scaler Berhasil Difit")

# Streamlit App
st.title("Prediksi Cluster Negara dengan K-Means")
st.write("Gunakan slider di bawah untuk memasukkan data dan memprediksi cluster negara:")

# Input Slider
user_input = {}
for col in columns:
    min_val = float(df_clean[col].min())
    max_val = float(df_clean[col].max())
    user_input[col] = st.slider(f"{col.capitalize()}", min_val, max_val, (min_val + max_val) / 2)

# Convert input to DataFrame
st.write("### Debug: Slider Input Selesai")
user_df = pd.DataFrame([user_input])

# Standardize input
user_scaled = scaler.transform(user_df)
st.write("### Debug: Input Data Berhasil Distandardisasi")

# Predict Cluster
cluster = kmeans.predict(user_scaled)[0]

# Output
st.write(f"### Hasil Prediksi: Negara ini berada di Cluster {cluster}")
