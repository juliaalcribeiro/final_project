# 🍷 PairVino: Wine–Food Pairing Recommender

PairVino is a data-driven, machine learning-powered application designed to provide expert wine recommendations for specific food pairings. It helps users overcome the complexity of wine knowledge by suggesting the **Top 5** best-paired wines based on the selected food item, category, cuisine, and a preferred wine category filter.

This project was built during the Ironhack Data Analytics Bootcamp as a final project.

---

## ✨ Project Highlights

* **Objective:** Build a robust recommender system to predict the optimal wine-food pairing quality (rated 1-5).
* **Detailed EDA:** Comprehensive Exploratory Data Analysis focused on understanding pairing performance across different wine colors (Red/White) and global cuisines.
* **Model:** **LightGBM Regressor** achieved the best performance with an **RMSE ≈ 0.28** and **MAE ≈ 0.16**.
* **Application:** A user-friendly web application developed using **Streamlit** for real-time recommendations.
* **Data:** Utilized a synthetic dataset from Kaggle ("Wine & Food Pairing") containing over 34,000 wine-food pairing ratings - https://www.kaggle.com/datasets/wafaaelhusseini/wine-and-food-pairing-dataset.
* **Presentation:** https://docs.google.com/presentation/d/1jraSTxcrbVBi2rNsUenmReCzRcXcZn7Qz3DfJ_GiamY/edit?usp=sharing

---

## 🛠️ Project Structure

The project is organized into a standard data science pipeline, with detailed Exploratory Data Analysis (EDA) focused on specific wine and cuisine categories:

| File/Folder | Description |
| :--- | :--- |
| `notebook/1_EDA_initial.ipynb` | Initial exploratory data analysis of the full dataset, focusing on overall wine rating distribution. |
| `notebook/2_EDA_red.ipynb` | **Focused EDA on Red Wines:** Analyzes and visualizes the best-paired Red Wines based on a custom combined score (Avg. Rating + 5-Star Share - 1-Star Share). |
| `notebook/3_EDA_white.ipynb` | **Focused EDA on White Wines:** Analyzes and visualizes the best-paired White Wines based on a custom combined score. |
| `notebook/4_EDA_cuisine.ipynb` | **Focused EDA on Cuisines:** Analyzes and visualizes the best-paired Cuisines with wine categories. |
| `notebook/5_1_data_preparation.ipynb` | Prepares the data for machine learning, including feature engineering (cross-features), One-Hot Encoding, and train/test splits. |
| `notebook/5_2_collaborative_filtering.ipynb` | Baseline recommendation model using Collaborative Filtering (Matrix Factorization). |
| `notebook/6_gradientboosting.ipynb` | Experimentation with `GradientBoostingRegressor`. |
| `notebook/7_xgboost.ipynb` | Experimentation with `XGBoostRegressor`. |
| `notebook/8_lightgbm.ipynb` | **Final model training and evaluation** using `LGBMRegressor` (selected model). |
| `notebook/lgb_pairing_model.joblib` | The trained **LightGBM model** saved for deployment. |
| `app.py` | The **Streamlit web application** code for real-time recommendations. |
| `PairVino.pptx` | Project presentation slides. |
| `config.yaml` | Configuration file to manage file paths and settings. |

---

## 🚀 Getting Started

### Prerequisites

You will need Python 3.8+ and the following libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `lightgbm`
* `xgboost`
* `streamlit`
* `PyYAML`
* `joblib`

# # 🚀 Running the Application

1. Ensure the Data and Model Are in Place

The raw dataset (e.g., wine_food_pairing.csv) must be accessible at the path defined in config.yaml.

The trained model file (lgb_pairing_model.joblib) must be located in the specified directory (e.g., inside the notebook folder or as defined in your configuration).

2. Launch the Streamlit App
streamlit run app.py

3. Access the Application

Once started, the app will automatically open in your browser, or you can access it manually at:

http://localhost:8501


# # 📊 Model Performance

The LightGBM Regressor was selected as the final model due to its superior performance in predicting the pairing_quality score (1–5).

P## Performance Comparison

| Model                              | RMSE   | MAE    |
|------------------------------------|--------|--------|
| Collaborative Filtering            | 1.3989 | 1.1765 |
| Gradient Boosting                  | 1.4685 | 1.2356 |
| XGBoost                            | 1.4823 | 1.2494 |
| **LightGBM (Final Model)**         | **≈ 0.28** | **≈ 0.16** |

# # 📌 Conclusion:
The very low RMSE and MAE achieved by LightGBM indicate high accuracy when predicting pairing scores, making the recommendations reliable and robust.


