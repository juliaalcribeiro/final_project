# ======================================================
# Wine Pairing Recommender - Streamlit App (Top 5 Wines)
# ======================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import yaml
import os

# ======================================================
# 1. Load dataset and model
# ======================================================

@st.cache_data
def load_data():
    """Load dataset from config.yaml relative to script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "../config.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Yaml configuration file not found at {yaml_path}")

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
    """Load trained LightGBM model safely."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../notebook/lgb_pairing_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    return model

# Load data and model
df = load_data()
model = load_model()

# Min/max for scaling predictions back
min_r = df["pairing_quality"].min()
max_r = df["pairing_quality"].max()

# ======================================================
# 2. Precompute mappings for factorization and target encoding
# ======================================================

cols_to_factor = ["food_item", "food_category", "cuisine", "wine_category", "wine_type"]

id_mappings = {}
te_mappings = {}
for col in cols_to_factor:
    _, ids = pd.factorize(df[col])
    id_mappings[col] = dict(zip(df[col], ids))
    te_mappings[col] = df.groupby(col)["pairing_quality"].mean().to_dict()

# ======================================================
# 3. Prepare dropdown lists
# ======================================================

food_items = sorted(df["food_item"].unique())
food_categories = sorted(df["food_category"].unique())
cuisines = sorted(df["cuisine"].unique())
wine_categories = sorted(df["wine_category"].unique())

# ======================================================
# 4. Feature preparation for prediction
# ======================================================

def prepare_features_for_wine(food_item, food_category, cuisine, wine_category, wine_type):
    """
    Convert a food + wine combination into model input features.
    Ensures numeric features and includes cross features.
    """
    row = pd.DataFrame([{
        "food_item": food_item,
        "food_category": food_category,
        "cuisine": cuisine,
        "wine_category": wine_category,
        "wine_type": wine_type
    }])

    # Factorize IDs
    for col in cols_to_factor:
        row[col + "_id"] = id_mappings[col].get(row.at[0, col], -1)

    # Target encoding
    for col in cols_to_factor:
        row[col + "_te"] = te_mappings[col].get(row.at[0, col], df["pairing_quality"].mean())

    # Cross features (hash to int for LightGBM)
    row["wine_cuisine_cross_id"] = int(hash(f"{wine_type}||{cuisine}") % 10000)
    row["wine_foodcat_cross_id"] = int(hash(f"{wine_type}||{food_category}") % 10000)

    feature_cols = [
        "food_item_id", "food_item_te",
        "food_category_id", "food_category_te",
        "cuisine_id", "cuisine_te",
        "wine_category_id", "wine_category_te",
        "wine_type_id", "wine_type_te",
        "wine_cuisine_cross_id", "wine_foodcat_cross_id"
    ]

    # Ensure numeric
    for col in feature_cols:
        row[col] = pd.to_numeric(row[col], errors='coerce')

    return row[feature_cols]

# ======================================================
# 5. Recommendation function (Top 5 Wines)
# ======================================================

def recommend_top_wines(food_item, food_category, cuisine, preferred_wine_category, top_n=5):
    """
    Returns top N wines for given food options.
    Scores all wines in dataset of the selected wine category.
    """
    # Filter wines by selected wine_category
    wines_subset = df[df["wine_category"] == preferred_wine_category].copy()

    predictions = []
    for idx, wine_row in wines_subset.iterrows():
        X = prepare_features_for_wine(
            food_item=food_item,
            food_category=food_category,
            cuisine=cuisine,
            wine_category=wine_row["wine_category"],
            wine_type=wine_row["wine_type"]
        )
        pred_scaled = model.predict(X)[0]
        pred = pred_scaled * (max_r - min_r) + min_r
        predictions.append(pred)

    wines_subset["predicted_score"] = predictions
    top_wines = wines_subset.sort_values("predicted_score", ascending=False).head(top_n)

    return top_wines[["wine_type", "wine_category", "predicted_score"]]

# ======================================================
# 6. Streamlit UI
# ======================================================

st.set_page_config(page_title="Wine Pairing Recommender", page_icon="🍷")
st.title("🍷 Wine Pairing Recommender")
st.write("Select your meal and get the top 5 wine recommendations!")

# Dropdowns
food_item = st.selectbox("Food item:", food_items)
food_category = st.selectbox("Food category:", food_categories)
cuisine = st.selectbox("Cuisine:", cuisines)
preferred_wine_category = st.selectbox("Preferred wine category:", wine_categories)

# Predict button
if st.button("Recommend Top 5 Wines"):
    top_wines = recommend_top_wines(
        food_item=food_item,
        food_category=food_category,
        cuisine=cuisine,
        preferred_wine_category=preferred_wine_category,
        top_n=5
    )

    st.subheader("🍷 Top 5 Wine Recommendations")
    st.dataframe(top_wines.reset_index(drop=True))
    st.success("Higher score means a better pairing with your food selection.")