import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime
import joblib
import os

# Initialize session state variables
if 'report_history' not in st.session_state:
    st.session_state.report_history = []
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Define cache file paths
MODEL_PATH = 'cached_model.joblib'
SCALER_PATH = 'cached_scaler.joblib'

# Define feature names globally
FEATURE_NAMES = [
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

# Define key priority features
KEY_PRIORITY_FEATURES = [
    'Carbon_Emission_Reduction',
    'Renewable_Energy_Use', 
    'Sustainable_Sourcing_Practices',
    'Worker_Safety_Incidents',
    'Gender_Pay_Gap'
]

def get_score_color(score):
    """Determine color based on overall score"""
    # Calculate average of key priority features
    key_priority_avg = np.mean([company_data[feature] for feature in KEY_PRIORITY_FEATURES])
    
    if key_priority_avg >= 75:
        score = max(80, score)  # Ensure score is at least 80
        return "green", "✅ Excellent SDG compliance score with strong performance in key priorities!"
    elif key_priority_avg < 45:
        score = min(40, score)  # Ensure score is at most 40
        return "red", "⚠️ Score indicates significant room for improvement, especially in key priority areas."
    else:
        score = max(40, min(80, score))  # Clamp score between 40 and 80
        return "yellow", "ℹ️ Score is average, improvements needed in key priority metrics."

def generate_training_data(n_samples=5000):  # Reduced from 10000 to 5000
    # Create more extreme cases in the synthetic data
    base_features = np.random.beta(0.3, 0.3, size=(n_samples, len(FEATURE_NAMES)))
    data = pd.DataFrame(base_features * 100, columns=FEATURE_NAMES)

    # Adjust binary features
    binary_features = ['Fair_Wages', 'Job_Creation', 'Worker_Safety', 'R&D_Investment',
                      'Infrastructure_Projects', 'Sustainable_Products', 'Waste_Management',
                      'Climate_Policies']
    
    data[binary_features] = (np.random.beta(0.2, 0.2, size=(n_samples, len(binary_features))) > 0.5).astype(int)

    # Generate target with more extreme distribution
    weights = np.random.uniform(1.0, 3.0, size=len(FEATURE_NAMES))
    # Give higher weights to key priority features
    for feature in KEY_PRIORITY_FEATURES:
        weights[FEATURE_NAMES.index(feature)] *= 2
        
    base_score = np.dot(data, weights)
    
    noise = np.random.normal(0, 10, n_samples)
    target = base_score + noise
    
    # Adjust scores based on key priority features
    key_priority_scores = data[KEY_PRIORITY_FEATURES].mean(axis=1)
    target = np.where(key_priority_scores >= 75, np.maximum(target * 1.5, 80), target)
    target = np.where(key_priority_scores < 45, np.minimum(target * 0.5, 40), target)
    target = np.where((key_priority_scores >= 45) & (key_priority_scores < 75), 
                     np.clip(target, 40, 80), target)
    
    target = np.clip(((target - target.min()) / (target.max() - target.min())) * 100, 0, 100)

    return data, target

def train_ml_model():
    # Check if cached model exists
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler
        except:
            pass  # If loading fails, train a new model

    # Generate training data with fewer samples
    X, y = generate_training_data(n_samples=5000)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model with fewer trees
    model = RandomForestRegressor(
        n_estimators=100,  # Reduced from 300 to 100
        max_depth=10,      # Reduced from 15 to 10
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1  # Use all CPU cores for faster training
    )
    model.fit(X_train_scaled, y_train)
    
    # Cache the model and scaler
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
    except:
        pass  # If saving fails, continue anyway
    
    return model, scaler

# Streamlit UI
st.title("🌍 SDG Tracker")
st.write("This app calculates a company's SDG (Sustainable Development Goals) compliance score based on key sustainability metrics.")

# Train ML model in sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This tool uses machine learning to evaluate SDG compliance.")
    
    if not st.session_state.model_trained:
        with st.spinner("Loading ML model..."):
            st.session_state.ml_model, st.session_state.scaler = train_ml_model()
            st.session_state.model_trained = True
        st.success("Model ready!")

# User Input Section
st.header("🏢 Enter Your Company Data")
company_name = st.text_input("Company Name", "Example Corp")

st.write("### 📊 Provide your company's sustainability metrics:")

feature_descriptions = {
    'Fair_Wages': "Does your company pay above industry average wages?",
    'Job_Creation': "Has your company created new jobs in the past year?",
    'Worker_Safety': "Do you have comprehensive worker safety protocols?",
    'R&D_Investment': "Do you invest in research and development?",
    'Infrastructure_Projects': "Are you implementing infrastructure improvement projects?",
    'Sustainable_Products': "Do you offer sustainable product alternatives?",
    'Waste_Management': "Do you have a waste management system?",
    'Climate_Policies': "Do you have climate action policies?",
    'Technology_Adoption': "Rate your company's adoption of new technologies (0-100)",
    'Circular_Economy_Practices': "Rate your implementation of circular economy practices (0-100)",
    'Carbon_Emission_Reduction': "Rate your carbon emission reduction efforts (0-100)",
    'Employment_Rate': "What is your current employment rate? (0-100)",
    'Digital_Literacy': "Rate your workforce's digital literacy level (0-100)",
    'Recycling_Rates': "What percentage of your waste is recycled? (0-100)",
    'Renewable_Energy_Use': "What percentage of your energy comes from renewable sources? (0-100)",
    'Climate_Risk_Assessment': "Rate your climate risk assessment implementation (0-100)",
    'Sustainable_Sourcing_Practices': "Rate your sustainable sourcing practices (0-100)",
    'Skills_Development_Programs': "Rate your skills development program effectiveness (0-100)",
    'Unemployment_Rate': "What is your company's unemployment rate? (0-100, lower is better)",
    'Youth_Unemployment_Rate': "What is your youth unemployment rate? (0-100, lower is better)",
    'Gender_Pay_Gap': "What is your gender pay gap? (0-100, lower is better)",
    'Worker_Safety_Incidents': "Rate of safety incidents (0-100, lower is better)",
    'Employee_Turnover_Rate': "What is your employee turnover rate? (0-100, lower is better)",
    'R&D_Expenditure': "Rate your R&D expenditure relative to industry average (0-100)",
    'Number_of_Patents': "Rate your patent filing activity relative to industry average (0-100)",
    'Infrastructure_Investment': "Rate your infrastructure investment level (0-100)",
    'Waste_Generation_Per_Capita': "Rate of waste generation (0-100, lower is better)",
    'Water_Consumption_Per_Unit': "Rate of water consumption (0-100, lower is better)",
    'Energy_Consumption_Per_Unit': "Rate of energy consumption (0-100, lower is better)",
    'Product_Lifecycles': "Rate your product lifecycle sustainability (0-100)",
    'GHG_Emissions': "Rate of greenhouse gas emissions (0-100, lower is better)",
    'Carbon_Footprint': "Rate your carbon footprint (0-100, lower is better)",
    'Energy_Consumption': "Rate your energy consumption efficiency (0-100)",
    'Mitigation_And_Adaptation_Strategies': "Rate your climate change mitigation strategies (0-100)",
    'Access_to_Reliable_Electricity': "Rate of access to reliable electricity (0-100)",
    'Minimum_Wage_Compliance': "Rate your minimum wage compliance (0-100)",
    'Adoption_of_New_Technologies': "Rate your adoption of new technologies (0-100)"
}

company_data = {}
for feature in FEATURE_NAMES:
    st.write(f"#### {feature.replace('_', ' ')}")
    st.write(f"*{feature_descriptions.get(feature, 'Rate your performance (0-100)')}*")
    
    if feature in ['Fair_Wages', 'Job_Creation', 'Worker_Safety', 'R&D_Investment',
                   'Infrastructure_Projects', 'Sustainable_Products', 'Waste_Management',
                   'Climate_Policies']:
        response = st.radio(f"Status", ["Yes", "No"], key=feature)
        company_data[feature] = 1 if response == "Yes" else 0
    else:
        default_value = 50.0
        if feature in KEY_PRIORITY_FEATURES:
            st.markdown("**🔑 KEY PRIORITY METRIC**")
            company_data[feature] = st.slider("Score", 0.0, 100.0, default_value, key=f"key_{feature}")
        else:
            company_data[feature] = st.slider("Score", 0.0, 100.0, default_value, key=feature)

company_df = pd.DataFrame([company_data])

# Prediction Section
if st.button("🔍 Calculate SDG Score"):
    if st.session_state.model_trained:
        # Get ML prediction
        company_scaled = st.session_state.scaler.transform(company_df)
        score = st.session_state.ml_model.predict(company_scaled)[0]

        # Adjust score based on key priority features average
        key_priority_avg = np.mean([company_data[feature] for feature in KEY_PRIORITY_FEATURES])
        if key_priority_avg >= 75:
            score = max(80, score)
        elif key_priority_avg < 45:
            score = min(40, score)
        else:
            score = max(40, min(80, score))

        st.success(f"🌟 **SDG Compliance Score for {company_name}: {score:.2f}**")

        # Feature importance analysis
        st.write("### Feature Importance Analysis")
        show_importance = st.checkbox("Show Feature Importance Analysis", key="show_importance")
        
        if show_importance:
            importances = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Importance': st.session_state.ml_model.feature_importances_
            })
            
            importances = importances.sort_values('Importance', ascending=True).tail(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            importances.plot(kind='barh', x='Feature', y='Importance', ax=ax)
            plt.title("Top 10 Most Important Features")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            st.pyplot(fig)
            
            st.write("### Detailed Feature Importance Scores")
            st.dataframe(importances)

        # Score visualization
        fig, ax = plt.subplots(figsize=(6, 3))
        bar_color, message = get_score_color(score)
        
        ax.barh(["SDG Score"], [score], color=bar_color)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score")
        ax.set_title(f"SDG Compliance Score for {company_name}")
        st.pyplot(fig)
        
        st.write(message)

        # Add download buttons for the data
        company_df['SDG_Score'] = score
        
        csv = company_df.to_csv(index=False)
        st.download_button(
            label="📥 Download data as CSV",
            data=csv,
            file_name=f'{company_name}_sdg_data.csv',
            mime='text/csv'
        )

        # Save to history
        report = {
            'timestamp': datetime.datetime.now(),
            'company_name': company_name,
            'score': score,
            'data': company_data
        }
        st.session_state.report_history.append(report)
    else:
        st.error("Please wait for the ML model to finish training.")

# Display History Section
st.header("📜 Report History")
if len(st.session_state.report_history) > 0:
    for i, report in enumerate(st.session_state.report_history):
        with st.expander(f"{report['company_name']} - {report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - Score: {report['score']:.2f}"):
            st.write("### Company Details:")
            report_df = pd.DataFrame([report['data']])
            st.dataframe(report_df)
            
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download this report",
                data=csv,
                file_name=f"{report['company_name']}_{report['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )

    if st.button("🗑️ Clear History"):
        st.session_state.report_history = []
        st.experimental_rerun()
else:
    st.info("No reports in history yet. Generate a prediction to create a report.")