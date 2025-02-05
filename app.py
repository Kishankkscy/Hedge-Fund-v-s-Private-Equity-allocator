import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Function to generate sample data
def generate_sample_data(n_samples=1000):
    feature_names = [
        'Fair_Wages', 'Job_Creation', 'Worker_Safety', 'R&D_Investment',
        'Infrastructure_Projects', 'Sustainable_Products', 'Waste_Management',
        'Climate_Policies', 'Technology_Adoption', 'Circular_Economy_Practices',
        'Carbon_Emission_Reduction', 'Employment_Rate', 'Digital_Literacy',
        'Recycling_Rates', 'Renewable_Energy_Use', 'Climate_Risk_Assessment',
        'Sustainable_Sourcing_Practices', 'Skills_Development_Programs',
        'Unemployment_Rate', 'Youth_Unemployment_Rate', 'Gender_Pay_Gap',
        'Worker_Safety_Incidents', 'Employee_Turnover_Rate', 'R&D_Expenditure',
        'Number_of_Patents', 'Infrastructure_Investment', 'Waste_Generation_Per_Capita',
        'Water_Consumption_Per_Unit', 'Energy_Consumption_Per_Unit', 'Product_Lifecycles',
        'GHG_Emissions', 'Carbon_Footprint', 'Energy_Consumption',
        'Mitigation_And_Adaptation_Strategies', 'Access_to_Reliable_Electricity',
        'Minimum_Wage_Compliance', 'Adoption_of_New_Technologies'
    ]

    data = pd.DataFrame(np.random.rand(n_samples, len(feature_names)), columns=feature_names)

    # Adjust binary features
    binary_features = ['Fair_Wages', 'Job_Creation', 'Worker_Safety', 'R&D_Investment',
                       'Infrastructure_Projects', 'Sustainable_Products', 'Waste_Management',
                       'Climate_Policies']
    
    data[binary_features] = (np.random.rand(n_samples, len(binary_features)) > 0.5).astype(int)

    # Generate target variable (SDG score)
    target = np.random.normal(70, 15, n_samples)
    target = np.clip(target, 0, 100)

    return data, target, feature_names

# Function to train the model
def train_model(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)

    return model, scaler, mse

# Streamlit UI
st.title("🌍 SDG Tracker AI Model")
st.write("This app predicts a company's SDG (Sustainable Development Goals) compliance score based on key sustainability metrics.")

# Generate sample data and train model
st.sidebar.header("🔄 Model Training")
st.sidebar.write("The model is trained on **synthetic data** with 1000 samples.")

data, target, feature_names = generate_sample_data()
model, scaler, mse = train_model(data, target)

st.sidebar.write(f"**Mean Squared Error:** {mse:.2f}")

# User Input Section
st.header("🏢 Enter Your Company Data")
company_name = st.text_input("Company Name", "Example Corp")

st.write("### 📊 Provide your company's sustainability metrics:")

company_data = {}
for feature in feature_names:
    if feature in ['Fair_Wages', 'Job_Creation', 'Worker_Safety', 'R&D_Investment',
                   'Infrastructure_Projects', 'Sustainable_Products', 'Waste_Management',
                   'Climate_Policies']:
        company_data[feature] = st.radio(f"{feature.replace('_', ' ')} (Yes/No)", [0, 1])
    else:
        company_data[feature] = st.slider(f"{feature.replace('_', ' ')}", 0.0, 100.0, 50.0)

company_df = pd.DataFrame([company_data])

# Prediction
if st.button("🔍 Predict SDG Score"):
    company_scaled = scaler.transform(company_df)
    score = model.predict(company_scaled)[0]

    st.success(f"🌟 **Predicted SDG Compliance Score for {company_name}: {score:.2f}**")

    # Visualization
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(["SDG Score"], [score], color="green")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score")
    ax.set_title(f"SDG Compliance Score for {company_name}")

    st.pyplot(fig)
