import streamlit as st
import pandas as pd
import joblib
import json
from food_matching import find_closest_component  # Import NLP function

# Load trained RandomForest model
best_rf = joblib.load("random_forest_model.pkl")

# Load feature names
with open("nutrition_columns.json", "r") as f:
    nutrition_columns = json.load(f)

# Load the estimated nutrient contributions from least squares regression
component_nutrition_matrix = pd.read_csv("component_nutrition_matrix.csv", index_col=0)

# Streamlit UI
st.title("🍏 Food Nutrition Density Predictor")
st.write("Enter a food name to find similar known components and predict its nutrition density.")

# User input
user_food = st.text_input("Enter a food name:", "")

if user_food:
    # Step 1: Find closest known components
    matched_components = find_closest_component(user_food)

    if matched_components:
        st.subheader("🔍 Matched Food Components")
        st.write(", ".join(matched_components))

        # Step 2: Extract estimated nutrient contributions from our binary matrix
        matched_df = component_nutrition_matrix.loc[matched_components, nutrition_columns]
        st.subheader("📊 Nutrient Contributions")
        st.dataframe(matched_df)

        # Step 3: Predict Nutrition Density for Each Component
        matched_X = matched_df[nutrition_columns]  # Extract relevant features
        predicted_density = best_rf.predict(matched_X)

        # Step 4: Display predictions for individual components
        st.subheader("⚡ Predicted Nutrition Density for Each Component")
        predictions_df = pd.DataFrame(
            {"Food Component": matched_components, "Predicted Nutrition Density": predicted_density}
        )
        st.dataframe(predictions_df)

        # 🔹 Step 5: Reconstruct the Whole Food
        combined_nutrient_values = matched_df.sum()  # Sum the estimated nutrient values for all matched components

        # Step 6: Predict Nutrition Density for the whole food
        combined_X = pd.DataFrame([combined_nutrient_values], columns=nutrition_columns)  # Reshape for model input
        whole_food_density = best_rf.predict(combined_X)[0]  # Get single prediction

        # Step 7: Display the combined Nutrition Density prediction
        st.subheader("🌟 Predicted Nutrition Density for Whole Food")
        st.write(f"**{user_food}: {whole_food_density:.2f}**")  # Display prediction

    else:
        st.error("❌ No similar food components found.")
