# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore
import joblib
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error
from typing import Optional, Tuple, Dict, Any

# --- Page Configuration ---
st.set_page_config(
    page_title="River Level Forecasting Engine",
    page_icon="🌊",
    layout="wide",
)

# --- Asset Loading ---
@st.cache_resource
def load_all_assets() -> Dict[str, Any]:
    """Load all models and scalers into a dictionary."""
    assets = {"models": {}, "scalers": {}}
    try:
        assets["models"]["multi_lstm"] = keras.models.load_model('best_multivariate_model.keras')
        assets["scalers"]["multi_lstm"] = joblib.load('multivariate_scaler.joblib')
        
        assets["models"]["uni_lstm"] = keras.models.load_model('best_univariate_model.keras')
        # NOTE: Using the multivariate scaler for the univariate model as a workaround
        assets["scalers"]["uni_lstm"] = joblib.load('multivariate_scaler.joblib')

        assets["models"]["transformer"] = keras.models.load_model('best_transformer_model.keras')
        assets["scalers"]["transformer"] = joblib.load('multivariate_scaler.joblib')

    except Exception as e:
        st.error(f"Error loading a model or scaler: {e}")
        st.info("Please ensure all model (.keras) and scaler (.joblib) files are present.")
        st.stop()
    
    # Quick check for XGBoost, which was requested but often missing
    try:
        assets["models"]["xgboost"] = joblib.load('best_xgboost_model.joblib')
        assets["scalers"]["xgboost"] = joblib.load('xgboost_scaler.joblib')
    except FileNotFoundError:
        st.warning("XGBoost model/scaler not found. It will be excluded from the comparison.")
        
    return assets

@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Load and prepare the dataset."""
    try:
        df = pd.read_csv('combined_dataset.csv', index_col='datetime', parse_dates=True)
        # Ensure the column order is the same as during training for multivariate models
        if isinstance(df, pd.DataFrame):
            df = df[['stage_m'] + [col for col in df.columns if col != 'stage_m']]
            return df  # type: ignore
        return None
    except FileNotFoundError:
        st.error("Error: 'combined_dataset.csv' not found.")
        return None

# --- Data & Model Processing ---
def create_sequences(data: np.ndarray, n_steps: int) -> np.ndarray:
    """Create time series sequences from input data."""
    X = []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :])
    return np.array(X)

@st.cache_data
def get_validation_predictions(
    _model: keras.Model, _scaler: Any, _df: pd.DataFrame, is_univariate: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and cache predictions for the entire validation set for a given model."""
    split_date_train_end = '2019-01-01'
    val_df = _df[_df.index > split_date_train_end].copy()
    
    n_steps = 7
    n_features = _df.shape[1]

    if is_univariate:
        target_data = _df[['stage_m']].values
        scaled_data = _scaler.transform(_df)
        scaled_target = scaled_data[:, 0].reshape(-1, 1)
        X_val = create_sequences(scaled_target, n_steps)
    else:
        scaled_data = _scaler.transform(_df)
        X_val = create_sequences(scaled_data, n_steps)

    predictions_scaled = _model.predict(X_val)
    
    total_predictions = predictions_scaled.size
    dummy_preds = np.zeros((total_predictions, n_features))
    dummy_preds[:, 0] = predictions_scaled.flatten()
    
    rescaled_preds = _scaler.inverse_transform(dummy_preds)
    
    if predictions_scaled.ndim > 1:
        predictions_original = rescaled_preds[:, 0].reshape(predictions_scaled.shape[0], predictions_scaled.shape[1])
    else:
        predictions_original = rescaled_preds[:, 0]

    pred_index = _df.index[n_steps:]
    if predictions_original.ndim == 1:
        columns = ['Day 1']
    else:
        columns = [f'Day {i+1}' for i in range(predictions_original.shape[1])]

    preds_df = pd.DataFrame(predictions_original, index=pred_index, columns=columns)  # type: ignore
    
    preds_df = preds_df[preds_df.index > split_date_train_end]
    val_data_actuals = val_df[val_df.index.isin(preds_df.index)]
    
    return val_data_actuals, preds_df  # type: ignore

def calculate_metrics_for_days(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Calculate R2, MSE, and RMSE for each forecast day."""
    metrics = []
    for day in range(y_pred.shape[1]):
        day_col = f'Day {day+1}'
        true_shifted = y_true['stage_m'].shift(-day)
        common_index = y_pred.index.intersection(true_shifted.index)
        
        if common_index.empty:
            continue
            
        pred_vals = y_pred.loc[common_index, day_col]
        true_vals = true_shifted.loc[common_index]
        
        # Remove NaN values before calculating metrics
        mask = ~(pd.isna(pred_vals) | pd.isna(true_vals))
        pred_vals_clean = pred_vals[mask]
        true_vals_clean = true_vals[mask]
        
        # Skip if no valid data points remain
        if len(pred_vals_clean) == 0 or len(true_vals_clean) == 0:
            continue

        r2 = r2_score(true_vals_clean, pred_vals_clean)
        mse = mean_squared_error(true_vals_clean, pred_vals_clean)
        rmse = np.sqrt(mse)
        metrics.append({"Forecast Day": day + 1, "R²": r2, "MSE": mse, "RMSE": rmse})
        
    return pd.DataFrame(metrics).set_index("Forecast Day")

# --- Main App Execution ---
assets = load_all_assets()
df = load_data()

st.title("🌊 French Broad River: Water Level Forecasting Engine")

if df is None:
    st.stop()

# --- UI Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📜 Project Overview & Data", 
    "📊 Model Performance Comparison", 
    "🔮 Live 'What-If' Forecast"
])

# --- Tab 1: Project Overview ---
with tab1:
    st.header("Project Summary")
    st.markdown("""
    This project exhaustively tests multiple machine learning architectures to find the optimal model for 7-day water level forecasting. 
    The analysis concludes that a **Multivariate LSTM model**, which processes time-series sequences natively, generally outperforms other tested architectures like Univariate LSTMs and Transformers.
    
    Use the tabs above to explore the historical data, compare model performances, and generate your own live forecasts.
    """)
    
    st.header("Historical River Water Levels (All Years)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df.index, y=df['stage_m'], mode='lines', name='Water Level (m)', line=dict(color='royalblue')))
    fig_hist.update_layout(
        title="French Broad River Water Levels",
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# --- Tab 2: Model Performance Comparison ---
with tab2:
    st.header("Comparing Forecasting Models")
    st.info("Select a model to see its performance on the unseen validation data (2019-2025).")

    model_options = {
        "Multivariate LSTM": "multi_lstm",
        "Univariate LSTM": "uni_lstm",
        "Transformer": "transformer",
    }
    if "xgboost" in assets["models"]:
        model_options["XGBoost"] = "xgboost"

    selected_model_name = st.selectbox("Choose a model to evaluate:", options=list(model_options.keys()), index=0)
    model_key = model_options[selected_model_name]

    is_uni = "uni_" in model_key
    actuals, predictions = get_validation_predictions(
        assets["models"][model_key], assets["scalers"][model_key], df, is_univariate=is_uni
    )

    st.subheader(f"Performance Metrics for: {selected_model_name}")
    metrics_df = calculate_metrics_for_days(actuals, predictions)
    st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0, color='lightgreen'), use_container_width=True)  # type: ignore

    st.subheader("Forecast vs. Actual Values on Validation Set")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=actuals.index, y=actuals['stage_m'], mode='lines', name='Actual Values', line=dict(color='blue')))
    fig_comp.add_trace(go.Scatter(x=predictions.index, y=predictions['Day 1'], mode='lines', name='Day 1 Forecast', line=dict(color='red', dash='dash')))
    if 'Day 7' in predictions.columns:
        fig_comp.add_trace(go.Scatter(x=predictions.index, y=predictions['Day 7'], mode='lines', name='Day 7 Forecast', line=dict(color='orange', dash='dot')))
    fig_comp.update_layout(
        title=f"{selected_model_name}: Forecast vs. Actuals",
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# --- Tab 3: Interactive 'What-If' Forecast ---
with tab3:
    st.header("🔮 Interactive Water Level Forecasting")
    st.info("Select a date to generate a live 7-day forecast using the champion **Multivariate LSTM** model.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Forecast Settings")
        
        min_dt = df.index.min()
        max_dt = df.index.max()

        min_date_val = pd.to_datetime('2000-01-08').date()
        if isinstance(min_dt, pd.Timestamp):
            min_date_val = (min_dt + pd.DateOffset(days=7)).date()
        
        max_date_val = pd.to_datetime('2025-01-01').date()
        if isinstance(max_dt, pd.Timestamp):
            max_date_val = max_dt.date()

        selected_date = st.date_input(
            "Select 'Today's' Date for Forecast:",
            value=pd.to_datetime('2024-01-15').date(),
            min_value=min_date_val,
            max_value=max_date_val
        )
        
        st.subheader("Model Input Data (Last 7 Days):")
        input_end = pd.to_datetime(selected_date) - pd.DateOffset(days=1)
        input_start = input_end - pd.DateOffset(days=6)
        input_df = df.loc[input_start:input_end]
        st.dataframe(input_df, use_container_width=True)
        
        generate_forecast = st.button("🚀 Generate 7-Day Forecast", type="primary", use_container_width=True)

    with col2:
        if generate_forecast:
            if len(input_df) == 7:
                with st.spinner("Generating forecast..."):
                    scaler = assets['scalers']['multi_lstm']
                    model = assets['models']['multi_lstm']
                    input_scaled = scaler.transform(input_df)
                    input_reshaped = np.reshape(input_scaled, (1, 7, input_df.shape[1]))
                    
                    pred_scaled = model.predict(input_reshaped)
                    dummy = np.zeros((pred_scaled.shape[1], df.shape[1]))
                    dummy[:, 0] = pred_scaled.flatten()
                    pred_unscaled = scaler.inverse_transform(dummy)[:, 0]
                    
                    forecast_dates = pd.date_range(start=selected_date, periods=7)
                    forecast_df = pd.DataFrame({'Forecast': pred_unscaled}, index=forecast_dates)
                    
                    actual_dates_end = pd.to_datetime(selected_date) + pd.DateOffset(days=6)
                    actual_df = df.loc[pd.to_datetime(selected_date):actual_dates_end, ['stage_m']]
                    actual_df.rename(columns={'stage_m': 'Actual'}, inplace=True)
                    
                    comparison_df = forecast_df.join(actual_df)
                    comparison_df['Actual'] = pd.to_numeric(comparison_df['Actual'], errors='coerce')
                    
                    st.subheader("Forecast vs. Actual Results")
                    st.dataframe(comparison_df.style.format({'Forecast': '{:.3f}', 'Actual': '{:.3f}'}), use_container_width=True)
                    
                    valid_comparison = comparison_df.dropna()
                    if not valid_comparison.empty:
                        st.subheader("Daily Performance Metrics")
                        daily_metrics = []
                        for i in range(len(valid_comparison)):
                            true = valid_comparison['Actual'].iloc[[i]]
                            pred = valid_comparison['Forecast'].iloc[[i]]
                            daily_metrics.append({
                                "Day": i + 1,
                                "R²": r2_score(true, pred),
                                "MSE": mean_squared_error(true, pred),
                                "RMSE": np.sqrt(mean_squared_error(true, pred))
                            })
                        daily_metrics_df = pd.DataFrame(daily_metrics).set_index("Day")
                        st.dataframe(daily_metrics_df.style.format("{:.3f}"), use_container_width=True)

                    st.subheader("📈 Forecast Visualization")
                    fig_live = go.Figure()
                    fig_live.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Forecast'], name='Forecast', line=dict(color='orange', width=3), mode='lines+markers'))
                    if not valid_comparison.empty:
                        fig_live.add_trace(go.Scatter(x=valid_comparison.index, y=valid_comparison['Actual'], name='Actual', line=dict(color='blue', width=2), mode='lines+markers'))
                    fig_live.update_layout(title="Live Forecast vs. Actual Water Levels", hovermode='x unified')
                    st.plotly_chart(fig_live, use_container_width=True)

            else:
                st.error("❌ Not enough historical data for this date.")
        else:
            st.info("👆 Click the button to generate a forecast.")