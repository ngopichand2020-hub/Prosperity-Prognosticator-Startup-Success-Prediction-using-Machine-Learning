import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

# Title
st.title("Prosperity Prognosticator")
st.write("Predict startup success using Machine Learning")

# Input fields
funding_total_usd = st.number_input("Funding Total USD", value=0.0)
funding_rounds = st.number_input("Funding Rounds", value=0.0)
relationships = st.number_input("Relationships", value=0.0)
milestones = st.number_input("Milestones", value=0.0)
avg_participants = st.number_input("Average Participants", value=0.0)
is_top500 = st.number_input("Is Top 500 (0 or 1)", value=0.0)
has_VC = st.number_input("Has Venture Capital (0 or 1)", value=0.0)
age_first_funding_year = st.number_input("Age First Funding Year", value=0.0)

# Predict button
if st.button("Predict"):

    # Create dataframe with same column names used in training
    input_data = pd.DataFrame([[

        funding_total_usd,
        funding_rounds,
        relationships,
        milestones,
        avg_participants,
        is_top500,
        has_VC,
        age_first_funding_year

    ]], columns=[

        'funding_total_usd',
        'funding_rounds',
        'relationships',
        'milestones',
        'avg_participants',
        'is_top500',
        'has_VC',
        'age_first_funding_year'

    ])

    # Make prediction
    prediction = model.predict(input_data)

    # Map numeric result to text
    status_map = {
        0: "Operating (Successful Startup)",
        1: "Closed (Failed Startup)",
        2: "Acquired (Highly Successful Startup)"
    }

    result = status_map.get(prediction[0], "Unknown")

    # Show result
    st.success(f"Prediction: {result}")
