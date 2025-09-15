import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model & Data
# ----------------------------
pipe = joblib.load("crop_pred.pkl")
df1 = pd.read_csv("crop_production.csv")
df1.dropna(inplace=True)

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Crop Production Predictor", page_icon="🌾", layout="centered")

# ----------------------------
# Title
# ----------------------------
st.title("🌾 Crop Production Predictor")
st.markdown("Predict crop production based on region, season, crop type, and cultivation area.")

# ----------------------------
# Input Section (Form)
# ----------------------------
with st.form("prediction_form"):
    st.subheader("📋 Enter Crop Details")

    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("Select State", df1["State_Name"].unique())
        season = st.selectbox("Select Season", df1["Season"].unique())
        year = st.number_input("Crop Year", 
                               min_value=int(df1["Crop_Year"].min()), 
                               max_value=int(df1["Crop_Year"].max()), 
                               step=1)
    with col2:
        district = st.selectbox("Select District", df1["District_Name"].unique())
        crop = st.selectbox("Select Crop", df1["Crop"].unique())
        area = st.number_input("Cultivation Area (in hectares)", min_value=1.0, step=1.0)

    # Submit button
    submitted = st.form_submit_button("🌱 Predict Production")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    input_data = pd.DataFrame(
        [[state, district, year, season, crop, area]],
        columns=["State_Name","District_Name","Crop_Year","Season","Crop","Area"]
    )
    prediction = pipe.predict(input_data)[0]

    st.success(f"✅ Predicted Production: **{prediction:,.2f} units**")
    st.balloons()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("✨ Built with [Streamlit](https://streamlit.io/) | Model: XGBoost Regressor")
