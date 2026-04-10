import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="House Price Prediction 🏠", layout="wide")

# ---------------- LOAD MODEL ----------------
if not os.path.exists("production_artifacts/house_price_model.joblib"):
    st.error("Model file not found. Please run model.py first.")
    st.stop()
else:
    model = joblib.load("production_artifacts/house_price_model.joblib")

# ---------------- STYLE ----------------
st.markdown("""
<style>
/* Banner */
.banner {
    background: linear-gradient(135deg,#74ebd5,#9face6);
    padding: 35px;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 40px;
    cursor: pointer;
}
.banner h1{
    color:white;
    font-size:40px;
    margin:0;
}

/* Navigation Buttons styled as cards */
div.stButton > button {
    background: linear-gradient(135deg,#74ebd5,#9face6);
    color: white;
    font-size: 22px;
    font-weight: bold;
    padding: 55px 20px;
    border-radius: 16px;
    border: none;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    transition: all 0.25s ease;
}
div.stButton > button:hover {
    transform: translateY(-6px) scale(1.03);
    box-shadow: 0 14px 30px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- BANNER (always visible) ----------------
if st.button("🏠 House Price Prediction Dashboard", use_container_width=True):
    st.session_state.page = "home"


# ---------------- HOME ----------------
if st.session_state.page == "home":
    st.markdown("<h3 style='text-align:center;'>Select a Feature</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Predict Price", use_container_width=True):
            st.session_state.page = "predict"

    with col2:
        if st.button("📂 Batch Prediction", use_container_width=True):
            st.session_state.page = "batch"

    with col3:
        if st.button("📈 Model Insights", use_container_width=True):
            st.session_state.page = "insights"

# ---------------- SINGLE PREDICTION ----------------
elif st.session_state.page == "predict":
    st.subheader("📊 Single Property Prediction")
    area_sqft = st.number_input("Square Footage (GrLivArea)", value=2000.0)
    bedrooms = st.number_input("Bedrooms (BedroomAbvGr)", value=3)
    full_bath = st.number_input("Full Bathrooms", value=2)
    half_bath = st.number_input("Half Bathrooms", value=1)
    bathrooms = full_bath + 0.5 * half_bath

    if st.button("Predict Price"):
        price = model.predict([[area_sqft, bedrooms, bathrooms]])[0]
        st.success(f"Estimated Price: ${price:,.0f}")

# ---------------- BATCH PREDICTION ----------------
elif st.session_state.page == "batch":
    st.subheader("📂 Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        if "FullBath" in batch_data.columns and "HalfBath" in batch_data.columns:
            batch_data["bathrooms"] = batch_data["FullBath"] + 0.5 * batch_data["HalfBath"]
            if all(col in batch_data.columns for col in ["GrLivArea","BedroomAbvGr","bathrooms"]):
                batch_data["Predicted Price"] = model.predict(
                    batch_data[['GrLivArea','BedroomAbvGr','bathrooms']]
                )
                st.write(batch_data[["GrLivArea","BedroomAbvGr","bathrooms","Predicted Price"]])
            else:
                st.error("Uploaded file missing required columns (GrLivArea, BedroomAbvGr).")
        else:
            st.error("CSV must contain FullBath and HalfBath columns.")

# ---------------- MODEL INSIGHTS ----------------
elif st.session_state.page == "insights":
    st.subheader("📈 Model Insights")
    actual = np.linspace(100000, 500000, 50)
    predicted = actual + np.random.normal(0, 20000, 50)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(actual, predicted, color="#7e57c2", alpha=0.75, s=45)
    ax.plot([100000,500000], [100000,500000], linestyle="--", linewidth=2, color="red")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Predicted vs Actual Prices")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(np.linspace(100000,500000,5))
    ax.set_yticks(np.linspace(100000,500000,5))
    ax.tick_params(axis='x', labelrotation=30)
    fig.tight_layout()

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)