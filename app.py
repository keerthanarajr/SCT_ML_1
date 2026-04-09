import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error

# Load dataset
data = pd.read_csv('train.csv')

# Selected features
features = ['GrLivArea','BedroomAbvGr','FullBath','HalfBath','OverallQual','YearBuilt']
X = data[features]
y = data['SalePrice']

# Train model
model = LinearRegression()
model.fit(X, y)

# Sidebar navigation
st.sidebar.title("🏡 Dashboard Modes")
mode = st.sidebar.radio("Choose Mode", 
                        ["House Price Prediction", 
                         "Investor Portfolio Valuation",  
                         "Feature Insights",  
                         "Scenario Simulator", 
                         "Model Evaluation"])

# -------------------------------
# House Price Prediction
# -------------------------------
if mode == "House Price Prediction":
    st.title("🏠 House Price Prediction")
    st.subheader("Predict house prices using ML model")

    area = st.slider("Living Area (sq ft)", 500, 5000, 2000)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    full_bath = st.slider("Full Bathrooms", 0, 5, 2)
    half_bath = st.slider("Half Bathrooms", 0, 5, 1)
    overallqual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    yearbuilt = st.slider("Year Built", 1900, 2020, 2000)

    input_df = pd.DataFrame([{
        'GrLivArea': area,
        'BedroomAbvGr': bedrooms,
        'FullBath': full_bath,
        'HalfBath': half_bath,
        'OverallQual': overallqual,
        'YearBuilt': yearbuilt
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")

# -------------------------------
# Investor Portfolio Valuation
# -------------------------------
elif mode == "Investor Portfolio Valuation":
    st.title("📂 Investor Portfolio Valuation")
    uploaded_file = st.file_uploader("Upload CSV with property details")

    if uploaded_file:
        portfolio = pd.read_csv(uploaded_file)
        portfolio['PredictedPrice'] = model.predict(portfolio[features])
        portfolio['PredictedPrice'] = portfolio['PredictedPrice'].apply(lambda x: f"${x:,.2f}")
        st.write("### Valuation Results")
        st.dataframe(portfolio)

        numeric_preds = portfolio['PredictedPrice'].str.replace('$','').str.replace(',','').astype(float)

        st.write("### Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${numeric_preds.sum():,.0f}")
        col2.metric("Average Value", f"${numeric_preds.mean():,.0f}")
        col3.metric("Properties", len(numeric_preds))

        st.download_button(
            label="Download Predictions",
            data=portfolio.to_csv(index=False).encode('utf-8'),
            file_name="portfolio_predictions.csv",
            mime="text/csv"
        )

# -------------------------------
# Feature Insights
# -------------------------------
elif mode == "Feature Insights":
    st.title("🔍 Feature Insights")
    importance = pd.Series(model.coef_, index=features)

    fig, ax = plt.subplots()
    importance.plot(kind='bar', color="#FFD966", ax=ax)
    ax.set_title("Feature Impact on Predicted Price")
    ax.set_ylabel("Coefficient Value")
    st.pyplot(fig)


# -------------------------------
# Scenario Simulator
# -------------------------------
elif mode == "Scenario Simulator":
    st.title("🤔 Scenario Simulator")

    scenario = st.selectbox("Choose Scenario", 
                            ["Custom", "Luxury Upgrade", "Budget Build", "Modern Renovation"])

    area = 2000
    bedrooms = 3
    full_bath = 2
    half_bath = 1
    overallqual = 5
    yearbuilt = 2000

    if scenario == "Luxury Upgrade":
        overallqual = 9
    elif scenario == "Budget Build":
        overallqual = 4
    elif scenario == "Modern Renovation":
        yearbuilt += 20

    input_df = pd.DataFrame([{
        'GrLivArea': area,
        'BedroomAbvGr': bedrooms,
        'FullBath': full_bath,
        'HalfBath': half_bath,
        'OverallQual': overallqual,
        'YearBuilt': yearbuilt
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Scenario '{scenario}' → Predicted Value: ${prediction:,.2f}")

# -------------------------------
# Model Evaluation
# -------------------------------
elif mode == "Model Evaluation":
    st.title("📈 Model Evaluation Metrics")
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    evs = explained_variance_score(y, y_pred)
    medae = median_absolute_error(y, y_pred)

    st.write("### Evaluation Results")
    st.write(f"- Mean Absolute Error (MAE): {mae:,.2f}")
    st.write(f"- Root Mean Squared Error (RMSE): {rmse:,.2f}")
    st.write(f"- R² Score: {r2:.3f}")
    st.write(f"- Explained Variance Score: {evs:.3f}")
    st.write(f"- Median Absolute Error: {medae:,.2f}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#999;'>Built by KEERTHANA • Internship Project</p>", unsafe_allow_html=True)