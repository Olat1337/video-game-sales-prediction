# 🎮 Global Video Game Sales Predictor

An end-to-end Machine Learning pipeline designed to predict the global sales of video games based on categorical features such as Platform, Genre, and Publisher.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.2+-150458.svg?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-2.1+-013243.svg?logo=numpy)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter)](https://jupyter.org/)

## 📊 Market Insights & Model Visualizations
<img width="1588" height="638" alt="image" src="https://github.com/user-attachments/assets/ef7fdff5-742b-4ea7-9fa0-b40dc0dc480a" />

## 🎯 Project Overview
Predicting how well a video game will sell is notoriously difficult. This project explores the gaming market by building a Machine Learning pipeline that estimates global sales based purely on pre-release categorical data.

This project was built to demonstrate a clean Data Science workflow-from preventing data leakage during preprocessing, to feature encoding, and finally comparing baseline linear models against advanced tree-based ensembles (Random Forest) to observe how they handle massive industry outliers.

## 🛠️ Technology Stack

To ensure rigorous analysis and model evaluation, the project utilizes the standard Python data science ecosystem.

### Data Engineering & EDA
* **Pandas & NumPy:** For data manipulation, handling missing values, and matrix transformations.
* **Seaborn & Matplotlib:** For generating visual market insights, comparing predictions against reality using log-scale error bands, and mapping numeric feature importance.

### Machine Learning
* **Scikit-Learn:** Built and tuned both a baseline Linear Regression model and a constrained Random Forest Regressor. 
* **Data Leakage Mitigation:** Ensured that post-release regional sales data (`NA_Sales`, `EU_Sales`) were strictly excluded from the training environment.

## 📈 Data & Modeling Workflow

The complete analysis and model training are documented in `video_game_sales_prediction.ipynb`:

* **Data Cleaning:** Removed identifiers lacking predictive power and dropped features that would cause data leakage.
* **Feature Encoding:** Converted text categories (e.g., "Action", "PS4", "Ubisoft") into a sparse numerical matrix using One-Hot Encoding.
* **Train/Test Split:** Isolated the `Global_Sales` target and split the data 80/20 to validate model generalization on unseen games.
* **Modeling:** Trained a baseline Linear Regression (with `np.clip` to prevent negative sales) and a Random Forest Regressor restricted to `max_depth=15` to capture non-linear relationships.
* **Overfitting Analysis:** Benchmarked the constrained Random Forest against an unrestricted depth model to analyze variance and memorization behaviors.

## 🧠 Key Technical & Visual Insights

* **The Blockbuster Ceiling:** As visualized in the log-scale scatter plot, the Random Forest is highly accurate for standard games (falling neatly within the ± 2x error margin band) but hits a hard predictive ceiling, completely failing to extrapolate the massive sales of hits (1M+ copies).
* **Nintendo's Market Power:** Feature importance analysis revealed that alongside the release `Year`, being published by `Nintendo` is the absolute strongest categorical driver of global sales, vastly outweighing specific game genres or console platforms.
* **Overfitting is Real:** The unrestricted Random Forest model memorized specific developer/platform combinations from the training data, ultimately failing to capture true underlying patterns and performing worse on the unseen test data.

## 🚀 How to Run Locally

If you would like to run this analysis on your own machine, follow these steps:

1. **Clone the repository:**
```bash
git clone https://github.com/Olat1337/video-game-sales-prediction.git
cd video-game-sales-prediction
```
2. **Create and activate a virtual environment:**
On macOS/Linux:

```Bash
python3 -m venv .venv
source .venv/bin/activate
```
On Windows:

```Bash
python -m venv .venv
.venv\Scripts\activate
```
3. **Install dependencies:**

```Bash
pip install -r requirements.txt
```
4. **Run the Notebook:**
Launch Jupyter to explore the data, view the models, and generate the dashboard graphs.

```Bash
jupyter notebook
```
