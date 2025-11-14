# ======================================================
# Wine Pairing Recommender - Streamlit App
# ======================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import yaml
import os

# ======================================================
# 1. Load model and dataset
# ======================================================

@st.cache_data
def load_data():
    """Load dataset from config.yaml relative to script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "../config.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Yaml configuration file not found at {yaml_path}")

    # Load YAML safely
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    csv_file = config.get('input_data', {}).get('file')
    if not csv_file:
        raise ValueError("CSV file path not found in config.yaml under 'input_data.file'")

    if not os.path.isabs(csv_file):
        csv_file = os.path.join(script_dir, csv_file)

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at {csv_file}")

    df = pd.read_csv(csv_file)
    return df

@st.cache_resource
def load_model():
    """Load the trained LightGBM model safely."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../notebook/lgb_pairing_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    return model

df = load_data()
model = load_model()

# These values must match those used during training
# If you saved min_r and max_r in a file, load them instead
min_r = df["pairing_quality"].min()
max_r = df["pairing_quality"].max()

# ======================================================
# 2. Prepare dropdown lists
# ======================================================

food_items = sorted(df["food_item"].unique())
food_categories = sorted(df["food_category"].unique())
cuisines = sorted(df["cuisine"].unique())
wine_categories = sorted(df["wine_category"].unique())
wine_types = sorted(df["wine_type"].unique())

# ======================================================
# 3. Feature preparation for prediction
# ======================================================

def prepare_features(food_item, food_category, cuisine, preferred_wine_category):
    """
    Convert user selections into model input features.
    This must replicate the feature engineering used during training.
    """

    # Create a single-row DataFrame
    row = pd.DataFrame([{
        "food_item": food_item,
        "food_category": food_category,
        "cuisine": cuisine,
        "wine_category": preferred_wine_category
    }])

    # === Factorize IDs (same order as training) ===
    for col in ["food_item", "food_category", "cuisine", "wine_category"]:
        row[col + "_id"] = pd.factorize(df[col])[0][ df[col] == row[col].iloc[0] ].iloc[0]

    # === Target encoding ===
    for col in ["food_item", "food_category", "cuisine", "wine_category"]:
        mapping = df.groupby(col)["pairing_quality"].mean()
        row[col + "_te"] = mapping[row[col].iloc[0]]

    # === Cross features ===
    row["wine_cuisine_cross_id"] = pd.factorize(
        df["wine_type"].astype(str) + "||" + df["cuisine"].astype(str)
    )[0][0]

    row["wine_foodcat_cross_id"] = pd.factorize(
        df["wine_type"].astype(str) + "||" + df["food_category"].astype(str)
    )[0][0]

    # Select feature columns
    feature_cols = [
        "food_item_id", "food_item_te",
        "food_category_id", "food_category_te",
        "cuisine_id", "cuisine_te",
        "wine_category_id", "wine_category_te",
        "wine_cuisine_cross_id", "wine_foodcat_cross_id"
    ]

    return row[feature_cols]

# ======================================================
# 4. Recommendation function
# ======================================================

def recommend_wines(food_item, food_category, cuisine, preferred_wine_category):
    """Use the trained model to generate predicted pairing score."""

    X = prepare_features(food_item, food_category, cuisine, preferred_wine_category)
    pred_scaled = model.predict(X)[0]

    # Convert scaled prediction back to original scale
    pred = pred_scaled * (max_r - min_r) + min_r
    return round(pred, 2)

# ======================================================
# 5. Streamlit UI
# ======================================================

st.set_page_config(page_title="Wine Pairing Recommender", page_icon="🍷")
st.title("🍷 Wine Pairing Recommender")
st.write("Choose your meal options and get the best wine recommendation.")

# --- Dropdown UI ---
food_item = st.selectbox("Food item:", food_items)
food_category = st.selectbox("Food category:", food_categories)
cuisine = st.selectbox("Cuisine:", cuisines)
preferred_wine_category = st.selectbox("Preferred wine category:", wine_categories)

# ======================================================
# 6. Predict button
# ======================================================

if st.button("Recommend Wine"):
    score = recommend_wines(
        food_item=food_item,
        food_category=food_category,
        cuisine=cuisine,
        preferred_wine_category=preferred_wine_category
    )

    st.subheader("🍷 Recommended Wine Pairing Score")
    st.metric(label="Predicted Quality", value=score)

    st.success("Higher score means a better wine pairing with your food selection.")
