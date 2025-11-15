# ======================================================
# Wine–Food Pairing Recommender – Streamlit App (Top-5)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import joblib

# ======================================================
# 1. Load configuration, dataset, and model
# ======================================================

@st.cache_data
def load_config_and_data():
    """Loads config.yaml and dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "../config.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"config.yaml not found at {yaml_path}")

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    csv_path = config["input_data"]["file"]

    if not os.path.isabs(csv_path):
        csv_path = os.path.join(script_dir, csv_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df0 = pd.read_csv(csv_path)
    return df0


@st.cache_resource
def load_model():
    """Loads the LightGBM trained model."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../notebook/lgb_pairing_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return joblib.load(model_path)


# Load data
df0 = load_config_and_data()
model = load_model()


# ======================================================
# 2. Rebuild identical feature engineering (training logic)
# ======================================================

target_col = "pairing_quality"

group_cols = [
    c for c in ["wine_type", "wine_category", "food_item",
                "food_category", "cuisine"]
    if c in df0.columns
]

# Aggregate dataset
df = df0.groupby(group_cols)[target_col].mean().reset_index()

# Scaling parameters
min_r, max_r = df[target_col].min(), df[target_col].max()
df["target_scaled"] = (df[target_col] - min_r) / (max_r - min_r)

# Category columns (same order as training)
cat_cols = group_cols.copy()

# Build feature columns used during model training
feature_cols = []
for c in cat_cols:
    feature_cols += [c + "_id", c + "_te"]
feature_cols += ["wine_cuisine_cross_id", "wine_foodcat_cross_id"]


# ----------------- Helper Functions --------------------

def factorize_and_map(df_full, df_query, col):
    """Map factorized IDs exactly as during training."""
    _, uniques = pd.factorize(df_full[col])
    mapping = {v: i for i, v in enumerate(uniques)}
    return df_query[col].map(mapping).fillna(-1).astype(int)


def target_encode(df_full, df_query, col):
    """Target encoding using mean target_scaled per category."""
    te_map = df_full.groupby(col)["target_scaled"].mean()
    global_mean = df_full["target_scaled"].mean()
    return df_query[col].map(te_map).fillna(global_mean)


def make_cross_for_query(df_full, df_query, colA, colB, new_col_name):
    """Cross-feature encoding exactly like training."""
    full_cross = (df_full[colA].astype(str) + "||" + df_full[colB].astype(str))
    _, uniques = pd.factorize(full_cross)
    mapping = {v: i for i, v in enumerate(uniques)}

    query_cross = (df_query[colA].astype(str) + "||" + df_query[colB].astype(str))
    df_query[new_col_name] = query_cross.map(mapping).fillna(-1).astype(int)
    return df_query


# ======================================================
# 3. Recommendation Function – Top 5 Wines
# ======================================================

def recommend_wines(food_item, food_category, cuisine, wine_category_filter=None):
    """Returns Top-5 wines with pairing_quality score."""

    # Candidate wines
    wines = df[["wine_type", "wine_category"]].drop_duplicates()

    if wine_category_filter:
        wines = wines[wines["wine_category"] == wine_category_filter]

    if wines.empty:
        raise ValueError("No wines available for this wine_category.")

    # Build query dataframe
    q = wines.copy()
    q["food_item"] = food_item
    q["food_category"] = food_category
    q["cuisine"] = cuisine

    # Apply feature engineering to the query
    for c in cat_cols:
        q[c + "_id"] = factorize_and_map(df, q, c)
        q[c + "_te"] = target_encode(df, q, c)

    q = make_cross_for_query(df, q, "wine_type", "cuisine", "wine_cuisine_cross_id")
    q = make_cross_for_query(df, q, "wine_type", "food_category", "wine_foodcat_cross_id")

    # Prepare input
    Xq = q[feature_cols]

    # Model prediction
    q["pred_scaled"] = model.predict(Xq)
    q["pred_score"] = q["pred_scaled"] * (max_r - min_r) + min_r

    # If the exact real score exists in dataset, use it
    merged = q.merge(
        df[group_cols + [target_col]],
        on=group_cols,
        how="left"
    )

    merged["final_score"] = merged[target_col].fillna(merged["pred_score"])

    # Top 5
    top5 = (
        merged[["wine_type", "wine_category", "final_score"]]
        .sort_values("final_score", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    return top5


# ======================================================
# 4. Streamlit Interface (capitalized dropdowns + nice list)
# ======================================================

st.set_page_config(page_title="Wine Pairing Recommender", page_icon="🍷")
st.title("🍷 Wine–Food Pairing – Top 5 Recommendations")
st.write("Select your food options and get the Top-5 wine recommendations.")

# Build ordered unique lists from df (stable order)
food_items_list = sorted(df["food_item"].unique(), key=lambda s: str(s).lower())
food_categories_list = sorted(df["food_category"].unique(), key=lambda s: str(s).lower())
cuisines_list = sorted(df["cuisine"].unique(), key=lambda s: str(s).lower())
wine_categories_list = sorted(df["wine_category"].unique(), key=lambda s: str(s).lower())

# Create display versions using title case (first letter of each word uppercase)
food_items_display = [str(s).title() for s in food_items_list]
food_categories_display = [str(s).title() for s in food_categories_list]
cuisines_display = [str(s).title() for s in cuisines_list]
wine_categories_display = [str(s).title() for s in wine_categories_list]

# Create reverse maps: display -> original (used for model input)
rev_food_item = dict(zip(food_items_display, food_items_list))
rev_food_category = dict(zip(food_categories_display, food_categories_list))
rev_cuisine = dict(zip(cuisines_display, cuisines_list))
rev_wine_category = dict(zip(wine_categories_display, wine_categories_list))

# --- User Inputs (dropdowns show Title Case) ---
food_item_cap = st.selectbox("Food Item:", food_items_display)
food_category_cap = st.selectbox("Food Category:", food_categories_display)
cuisine_cap = st.selectbox("Cuisine:", cuisines_display)
wine_category_cap = st.selectbox("Wine Category (Preference):", wine_categories_display)

# Map back to original values expected by the model
food_item = rev_food_item[food_item_cap]
food_category = rev_food_category[food_category_cap]
cuisine = rev_cuisine[cuisine_cap]
wine_category_filter = rev_wine_category[wine_category_cap]

# --- Button ---
if st.button("Recommend Wines"):
    results = recommend_wines(
        food_item=food_item,
        food_category=food_category,
        cuisine=cuisine,
        wine_category_filter=wine_category_filter
    )

    st.subheader("🍷 Top-5 Recommended Wines")

    # Output as stylized, numbered list (1 = best)
    for i, row in results.iterrows():
        rank = i + 1
        wine_type = str(row["wine_type"]).title()
        wine_category = str(row["wine_category"]).title()
        score = round(row["final_score"], 3)

        st.markdown(
            f"""
            <div style="padding:12px; border-radius:8px; background:linear-gradient(90deg,#fff,#f7fbff);">
              <h3 style="margin:0">{rank}. 🥂 <span style="color:#7b2cbf">{wine_type}</span> <small style="color:#666">({wine_category})</small></h3>
              <p style="margin:6px 0 0 0">⭐ <strong>Pairing Quality:</strong> {score}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")  # small spacing

    st.success("Higher pairing_quality means a better wine–food match.")