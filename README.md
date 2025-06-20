# Machine Learning Approaches for French Broad River Water Level Forecasting

**Author:** Muhammed Yasir Güneş  
**Supervisors:** Prof. Yusuf Sinan Akgül, Prof. Mahir İnce  
**Institution:** Gebze Technical University, Department of Computer Engineering  
**Project Date:** 2025

---

## 1. Project Overview

This project presents a comprehensive, end-to-end solution for 7-day water level forecasting on the French Broad River at USGS gauge 03443000. The primary goal was to conduct a rigorous comparative analysis of various machine learning architectures to identify the most effective and practical model for this real-world hydrological task.

The project encompasses a full pipeline: from automated data ingestion and preprocessing to model training, hyperparameter tuning, and evaluation. The key finding is that while state-of-the-art deep learning models like LSTMs perform well, the **XGBoost model provides the optimal balance of predictive accuracy, computational efficiency, and practical utility.**

To translate these findings into a tangible solution, the project culminates in an **interactive decision-support dashboard** built with Streamlit. This application provides live forecasts, robust confidence intervals, and visualizes predictions against official USGS flood alert levels, turning complex data into actionable insights.

[![Project Demo Video](https://img.youtube.com/vi/[PLACEHOLDER_YOUTUBE_VIDEO_ID]/0.jpg)](https://www.youtube.com/watch?v=[PLACEHOLDER_YOUTUBE_VIDEO_ID])

> **Click the image above to watch the 90-second project trailer on YouTube.**

---

## 2. Key Features

- **Automated Data Pipeline:** The system automatically fetches the latest data from USGS (water levels) and Meteostat (weather) to ensure forecasts are always current.
- **Rigorous Model Comparison:** Implements and evaluates four distinct models:
    1.  **Naive Persistence Baseline:** Forecasts future values based on the last known value.
    2.  **Multi-Layer Perceptron (MLP):** A standard feedforward neural network.
    3.  **LSTM Encoder-Decoder:** A state-of-the-art deep learning architecture for sequence-to-sequence tasks.
    4.  **XGBoost:** A powerful gradient-boosted decision tree ensemble.
- **Hyperparameter Tuning:** Utilizes `KerasTuner` to systematically find the optimal hyperparameters for the neural network models.
- **Interactive Dashboard:** A user-friendly web application built with Streamlit that includes:
    -   Real-time, on-demand 7-day forecasts.
    -   **Empirical Confidence Intervals:** Uncertainty is quantified based on the model's historical error distribution, not just statistical theory.
    -   **USGS Flood Level Integration:** Forecasts are visually contextualized with official flood alert bands ("Action", "Minor", "Moderate", "Major").
    -   In-depth performance analysis tabs to view model metrics and historical predictions.

---

## 3. Directory Structure

```
/french-broad-river-forecast/
|
|-- data/
|   |-- combined_dataset.csv         # The primary dataset used by the app
|
|-- models/
|   |-- best_baseline_mlp_model.keras  # Trained Keras MLP model
|   |-- best_baseline_xgboost_model.json # Trained XGBoost model
|   |-- best_multivariate_combined_model.keras # Trained Keras LSTM model
|   |-- multivariate_combined_scaler.joblib # The scaler fit on the training data
|
|-- notebooks/
|   |-- 01_data_preparation.ipynb    # Exploratory notebook for data fetching/cleaning
|   |-- 02_model_training.ipynb      # Exploratory notebook for model training/tuning
|
|-- app.py                         # The main Streamlit application script
|-- train.py                       # Consolidated script for training all models
|-- update_dataset.py              # Script to fetch latest data
|
|-- requirements.txt               # Required Python packages for the project
|-- README.md                      # This file
```

---

## 4. Setup and Installation

To run this project locally, please follow these steps. It is recommended to use a virtual environment.

**Step 1: Clone the repository**
```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/[YOUR_REPO_NAME].git
cd [YOUR_REPO_NAME]
```

**Step 2: Create and activate a virtual environment**
```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

**Step 3: Install the required packages**
```bash
pip install -r requirements.txt
```

---

## 5. How to Run the Project

The project has two main components: the data update script and the interactive dashboard.

### 5.1. Initial Data Setup

The Streamlit application includes a feature to automatically update the dataset. The first time you run the application, it will detect that `combined_dataset.csv` is missing and run the `update_dataset.py` script to generate it. This may take a few minutes.

### 5.2. Running the Interactive Dashboard

Once the setup is complete, launch the Streamlit application from your terminal:

```bash
streamlit run app.py
```

Your web browser will open, and you can begin interacting with the dashboard.

### 5.3. (Optional) Manual Model Retraining

If you wish to retrain the models from scratch, you can use the `train.py` script. This script will regenerate the model files located in the `/models` directory.

```bash
# To train a specific model
python train.py --model_type xgboost

# Available model types: baseline, mlp, lstm, xgboost, all
python train.py --model_type all
```

---

## 6. Conclusion and Key Findings

The comprehensive analysis revealed that the **XGBoost model provides the optimal solution** for this forecasting task. It achieved a 1-day ahead R² score of **0.903**, slightly outperforming the more complex LSTM model while requiring significantly less computational time for training.

This project successfully demonstrates a complete pipeline from academic research to a practical engineering solution, providing a valuable tool for local water resource management and flood risk awareness.