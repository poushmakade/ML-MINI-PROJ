import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Fuel Efficiency Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #0078ff;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #005fcc;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load('best_model.joblib')
except Exception as e:
    st.error("⚠️ Error loading model file. Please make sure 'best_model.joblib' is in the same folder.")
    st.stop()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🧮 Enter Vehicle Details")

cylinders = st.sidebar.number_input("Cylinders", min_value=3, max_value=16, value=4)
displacement = st.sidebar.number_input("Engine Displacement (liters)", min_value=0.5, max_value=10.0, value=2.0)
fuelCost08 = st.sidebar.number_input("Annual Fuel Cost (USD)", min_value=500, max_value=6000, value=2000)
model_year = st.sidebar.number_input("Model Year", min_value=1980, max_value=2025, value=2020)

fuelType1 = st.sidebar.selectbox("Fuel Type", ["Regular Gasoline", "Premium Gasoline", "Diesel", "Electricity", "CNG"])
trany = st.sidebar.selectbox("Transmission", ["Automatic", "Manual", "Auto (AM-S)", "CVT"])
VClass = st.sidebar.selectbox("Vehicle Class", ["Compact Cars", "Midsize Cars", "Large Cars", "SUV", "Pickup Trucks"])
drive = st.sidebar.selectbox("Drive Type", ["FWD", "RWD", "4WD", "AWD"])

# -----------------------------
# MAIN PAGE HEADER
# -----------------------------
st.markdown("<h1 style='text-align: center;'>🚗 Fuel Efficiency Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#bfbfbf;'>Predict vehicle fuel economy (MPG) using engine and design features.</p>", unsafe_allow_html=True)
st.write("---")

# -----------------------------
# PREDICTION
# -----------------------------
input_data = pd.DataFrame({
    'cylinders': [cylinders],
    'displacement': [displacement],
    'fuelType1': [fuelType1],
    'trany': [trany],
    'VClass': [VClass],
    'drive': [drive],
    'fuelCost08': [fuelCost08],
    'model_year': [model_year]
})

if st.button("🔍 Predict Fuel Efficiency"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Fuel Efficiency: {prediction[0]:.2f} MPG")
    except Exception as e:
        st.error("⚠️ Could not make prediction. Ensure input format matches training data.")

# -----------------------------
# ABOUT PROJECT SECTION
# -----------------------------
st.write("---")
st.subheader("📘 About the Project")
st.markdown("""
**Goal:**  
Estimate a vehicle’s **fuel efficiency (MPG)** using its engine and design parameters such as cylinders, engine displacement, transmission type, fuel type, and model year.

**Dataset:**  
*US Vehicle Fuel Economy Data* — containing real-world vehicle specs and performance metrics.

**Tech Stack:**  
-  Machine Learning (Scikit-Learn, XGBoost)  
-  Data Preprocessing with Pandas & Pipelines  
-  GUI built using Streamlit  

**How it works:**  
1. User enters vehicle specifications in the sidebar.  
2. Model processes the data through trained preprocessing pipelines.  
3. Predicted fuel efficiency (MPG) is displayed instantly.  

**Developed By:** Poush Makade  
*For Mini ML Project Submission*
""")

# -----------------------------
# FOOTER
# -----------------------------
st.write("---")
st.caption("© 2025 Fuel Efficiency Estimator | Mini Project | Developed using Streamlit")
