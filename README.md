# Video Game Sales Prediction Pipeline 🎮📊

## 📌 Overview
This project is an end-to-end Machine Learning pipeline designed to predict the global sales of video games based on categorical features such as Platform, Genre, and Publisher. The primary objective is to establish a regression baseline, evaluate model performance, and analyze how decision tree ensembles handle heavy outliers (e.g., blockbuster hits).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (LinearRegression, RandomForestRegressor)

## ⚙️ Project Workflow
1. **Data Preprocessing:** Cleaned the raw dataset by removing missing values and dropping features causing data leakage (e.g., regional sales like `NA_Sales`, `EU_Sales`).
2. **Feature Engineering:** Transformed text-based categorical variables into a sparse numerical matrix using One-Hot Encoding (`pd.get_dummies`).
3. **Modeling & Evaluation:** * Trained a baseline **Linear Regression** model.
   * Trained a **Random Forest Regressor** with restricted depth (`max_depth=15`) to capture non-linear relationships without overfitting.
   * Applied target clipping (`np.clip`) to prevent mathematically possible but physically impossible negative sales predictions.

## 📊 Key Insights & Results
* **Baseline vs. Ensemble:** The Random Forest achieved a better Mean Absolute Error (MAE), predicting "average" games more accurately. However, Linear Regression handled extreme outliers better, resulting in a lower Root Mean Squared Error (RMSE).
* **Extrapolation Limits:** Decision trees inherently struggle to predict values higher than the maximum seen in the training set. Consequently, the Random Forest heavily underestimated mega-hits like *GTA V* or *Wii Sports*.
* **Feature Limitations:** The current feature set (Platform, Genre, Publisher) lacks sufficient predictive power to confidently identify a blockbuster. Future iterations require data enrichment (e.g., marketing budgets, Metacritic review scores).

## 🚀 How to Run Locally

If you want to run this pipeline on your local Linux/macOS machine, follow these steps in your terminal:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Olat1337/video-game-sales-prediction.git
   cd video-game-sales-prediction
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the pipeline:**
  Open the Jupyter Notebook (video_game_sales_prediction.ipynb) or execute the Python script to view the model training and evaluation process.
