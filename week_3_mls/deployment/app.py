import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase a tourism package
based on their demographic and interaction details. Enter the customer information below to get a prediction.
""")

# Input features for the tourism dataset
# Numerical features
age = st.number_input("Age", 18, 100, 35)
city_tier = st.number_input("City Tier (1, 2, or 3)", 1, 3, 2)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 0, 60, 10)
num_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
num_followups = st.number_input("Number of Follow-ups", 0, 10, 3)
pref_property_star = st.number_input("Preferred Property Star (1-5)", 1, 5, 3)
num_trips = st.number_input("Number of Trips (yearly)", 0, 50, 5)
passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", 1, 5, 3)
own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
num_children_visiting = st.number_input("Number of Children Visiting", 0, 5, 0)
monthly_income = st.number_input("Monthly Income", 0.0, 200000.0, 50000.0)

# Categorical features
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited", "Direct Contact"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female", "Not disclosed"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Director"])

input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "NumberOfPersonVisiting": num_person_visiting,
    "NumberOfFollowups": num_followups,
    "PreferredPropertyStar": pref_property_star,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children_visiting,
    "MonthlyIncome": monthly_income,
    "TypeofContact": type_of_contact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status,
    "Designation": designation,
}])

if st.button("Predict Purchase"): # Changed button text
    prediction = model.predict(input_data)[0]
    result = "Will purchase tourism package" if prediction == 1 else "Will not purchase tourism package" # Updated prediction message
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
