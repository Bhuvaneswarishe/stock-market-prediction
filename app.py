import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the models
intel_model = load_model('intel_stock_model.h5')
amd_model = load_model('amd_stock_model.h5')

# Initialize scaler
scaler = MinMaxScaler()

# Streamlit App
st.title("Stock Price Prediction Dashboard")
st.sidebar.header("Upload Stock Data")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.dataframe(data.head())

    # Preprocess data
    st.sidebar.subheader("Preprocessing")
    time_steps = st.sidebar.slider("Select Time Steps", min_value=30, max_value=100, value=60, step=5)
    
    # Ensure 'Close' column exists
    if 'Close' not in data.columns:
        st.error("The uploaded file must contain a 'Close' column!")
    else:
        close_prices = data['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(close_prices)

        # Create sequences
        X = []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i, 0])
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Choose model
        model_choice = st.sidebar.selectbox("Select Model", ["Intel", "AMD"])
        model = intel_model if model_choice == "Intel" else amd_model

        # Predictions
        predictions = model.predict(X)
        predictions_rescaled = scaler.inverse_transform(predictions)

        # Visualize predictions
        actual = close_prices[time_steps:]
        st.write("### Prediction Results")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(actual, label="Actual Prices", color="blue")
        ax.plot(predictions_rescaled, label="Predicted Prices", color="red", alpha=0.7)
        ax.set_title(f"{model_choice} Stock Price Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Download predictions
        predictions_df = pd.DataFrame({"Actual": actual.flatten(), "Predicted": predictions_rescaled.flatten()})
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv, file_name=f"{model_choice}_predictions.csv")
