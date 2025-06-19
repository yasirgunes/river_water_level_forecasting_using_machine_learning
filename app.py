import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model as load_keras_model
from datetime import datetime, date
from scipy import stats
import subprocess
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="River Level Forecast",
    page_icon="🌊",
    layout="wide"
)

# --- Dataset Update Functions ---

def check_dataset_freshness():
    """
    Check if the dataset is up to date.
    Returns: (is_current, last_date, today_date)
    """
    try:
        if os.path.exists('combined_dataset.csv'):
            df = pd.read_csv('combined_dataset.csv', parse_dates=['datetime'], index_col='datetime')
            last_date = df.index.max().date()
            today = date.today()
            
            # Dataset is current if it includes yesterday's data (accounting for USGS 1-day lag)
            yesterday = pd.Timestamp.now().date() - pd.Timedelta(days=1)
            is_current = last_date >= yesterday
            
            return is_current, last_date, today
        else:
            return False, None, date.today()
    except Exception as e:
        st.error(f"Error checking dataset: {e}")
        return False, None, date.today()

def update_dataset_if_needed():
    """
    Check if dataset needs updating and update if necessary.
    Returns True if data was updated, False otherwise.
    """
    is_current, last_date, today = check_dataset_freshness()
    
    if is_current:
        return False
    
    # Show update status with more informational messaging
    if last_date:
        st.info(f"📊 Dataset will be updated to latest available data. Current data: {last_date}")
    else:
        st.info("📊 No dataset found. Creating new dataset...")
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Updating dataset with latest water level data...")
        progress_bar.progress(20)
        
        # Run the update script
        result = subprocess.run(['python', 'update_dataset.py'], 
                              capture_output=True, text=True, timeout=300)
        
        progress_bar.progress(80)
        
        if result.returncode == 0:
            progress_bar.progress(100)
            status_text.text("✅ Dataset updated successfully!")
            st.success("🎉 Dataset updated with latest available data!")
            
            # Clear cache to reload new data
            st.cache_data.clear()
            
            return True
        else:
            st.error(f"❌ Failed to update dataset: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        st.error("⏰ Dataset update timed out. Please try again later.")
        return False
    except Exception as e:
        st.error(f"❌ Error updating dataset: {e}")
        return False
    finally:
        progress_bar.empty()
        status_text.empty()

# --- Helper Functions ---

@st.cache_resource
def load_models_and_scalers():
    """Load all pre-trained models and scalers."""
    models = {}
    scalers = {}
    try:
        models['XGBoost'] = joblib.load('models/best_baseline_xgboost_model.joblib')
        scalers['XGBoost'] = joblib.load('models/multivariate_combined_scaler.joblib')
    except Exception:
        models['XGBoost'], scalers['XGBoost'] = None, None
    
    try:
        # MLP uses multivariate combined scaler (from baseline training)
        models['MLP'] = load_keras_model('models/best_baseline_mlp_model.keras')
        scalers['MLP'] = joblib.load('models/multivariate_combined_scaler.joblib')
    except Exception:
        models['MLP'], scalers['MLP'] = None, None

    try:
        # LSTM uses multivariate combined scaler and model
        models['LSTM'] = load_keras_model('models/best_multivariate_combined_model.keras')
        scalers['LSTM'] = joblib.load('models/multivariate_combined_scaler.joblib')
    except Exception:
        models['LSTM'], scalers['LSTM'] = None, None
        
    return models, scalers

@st.cache_data
def load_data():
    """Load the combined dataset."""
    try:
        # Try loading from root directory first (where update_dataset.py saves it)
        df = pd.read_csv('combined_dataset.csv', index_col='datetime', parse_dates=True)
        return df
    except FileNotFoundError:
        try:
            # Fallback to data directory
            df = pd.read_csv('data/combined_dataset.csv', index_col='datetime', parse_dates=True)
            return df
        except FileNotFoundError:
            return None

@st.cache_data
def calculate_prediction_errors(model_name, _model, _scaler, _df, n_past=7, n_future=7):
    """
    Calculate historical prediction errors for confidence interval estimation.
    Returns error statistics for each forecast day.
    """
    # Use dynamic validation split - from training start to today
    split_date_start = '2019-01-01'  # Keep training split as is
    today = pd.Timestamp.now().normalize()  # Get today's date at midnight
    
    # Use data from training split start to today (or dataset end, whichever is earlier)
    dataset_end = pd.Timestamp(str(_df.index.max())[:19])
    validation_end = min(today, dataset_end)
    
    val_data = _df[(_df.index > split_date_start) & (_df.index <= validation_end)]
    
    # Scale the validation data
    df_scaled_val = pd.DataFrame(_scaler.transform(val_data[_scaler.feature_names_in_]), 
                                columns=_scaler.feature_names_in_, 
                                index=val_data.index)
    
    # Create sequences for the entire validation set
    X_val, y_val = create_sequences(df_scaled_val, n_past, n_future)
    
    if len(X_val) == 0:
        return None
    
    # Get predictions based on model type
    if model_name == 'XGBoost':
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        y_pred_scaled = _model.predict(X_val_flat)
    elif model_name == 'MLP':
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        y_pred_scaled = _model.predict(X_val_flat)
    else:  # LSTM
        y_pred_scaled = _model.predict(X_val)
        if y_pred_scaled.ndim == 3:
            y_pred_scaled = y_pred_scaled.squeeze(axis=-1)
    
    # Inverse transform to get original scale
    n_features = len(_scaler.feature_names_in_)
    
    # Inverse transform predictions
    dummy_pred = np.zeros((len(y_pred_scaled.flatten()), n_features))
    dummy_pred[:, 0] = y_pred_scaled.flatten()
    y_pred_original = _scaler.inverse_transform(dummy_pred)[:, 0].reshape(y_pred_scaled.shape)
    
    # Inverse transform actual values
    dummy_actual = np.zeros((len(y_val.flatten()), n_features))
    dummy_actual[:, 0] = y_val.flatten()
    y_actual_original = _scaler.inverse_transform(dummy_actual)[:, 0].reshape(y_val.shape)
    
    # Calculate errors for each forecast day
    errors_by_day = {}
    error_stats = {}
    
    for day in range(n_future):
        errors = y_pred_original[:, day] - y_actual_original[:, day]
        errors_by_day[f'day_{day+1}'] = errors
        
        # Calculate error statistics
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        
        # Calculate confidence intervals (assuming normal distribution)
        # 68% CI (±1 std), 95% CI (±1.96 std), 99% CI (±2.58 std)
        error_stats[f'day_{day+1}'] = {
            'mean': error_mean,
            'std': error_std,
            'ci_68': 1.0 * error_std,    # 68% confidence interval
            'ci_95': 1.96 * error_std,   # 95% confidence interval
            'ci_99': 2.58 * error_std    # 99% confidence interval
        }
    
    return error_stats

def create_sequences(data, n_past, n_future):
    """Create sequences for LSTM/MLP models."""
    X, y = [], []
    numpy_data = data.values
    for i in range(len(numpy_data) - n_past - n_future + 1):
        X.append(numpy_data[i : i + n_past])
        y.append(numpy_data[i + n_past : i + n_past + n_future, 0])
    return np.array(X), np.array(y)

# --- Auto-Update Dataset on App Load ---
st.title("🌊 French Broad River Water Level Prediction")
st.write("Interactive dashboard for visualizing and predicting river water levels with confidence intervals.")

# Check and update dataset if needed
with st.spinner("🔍 Checking dataset freshness..."):
    dataset_updated = update_dataset_if_needed()

# Load resources after potential update
models, scalers = load_models_and_scalers()
full_df = load_data()

# Show dataset status
if full_df is not None:
    is_current, last_date, today = check_dataset_freshness()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📅 Latest Data", str(last_date) if last_date else "N/A")
    with col2:
        st.metric("📈 Total Records", f"{len(full_df):,}")

# --- Sidebar ---
with st.sidebar:
    st.title("About")
    st.write("This dashboard visualizes river water level data and predictions.")
    st.header("Features")
    st.markdown("- Historical data visualization\n- Water level forecasting\n- Confidence intervals\n- Model performance metrics\n- **Auto-updating dataset**")
    
    st.header("System Status")
    for model_name in ['XGBoost', 'MLP', 'LSTM']:
        if models.get(model_name) and scalers.get(model_name):
            st.success(f"{model_name} Model: Loaded")
        else:
            st.error(f"{model_name} Model: Not Loaded")

    if full_df is not None:
        st.success("Data: Loaded")
        
        # Add manual update button
        if st.button("🔄 Force Update Dataset"):
            st.cache_data.clear()  # Clear cache first
            update_dataset_if_needed()
            st.rerun()
    else:
        st.error("Data: Not Found")

if full_df is None:
    st.error("Dataset not found. Please check your internet connection and try refreshing the page.")
    st.stop()

with st.expander("View Raw Data"):
    st.dataframe(full_df.tail(100).style.format("{:.2f}"))

# --- Historical Data Section ---
st.header("Historical Water Level Data")

if not isinstance(full_df.index, pd.DatetimeIndex):
    st.error("The dataframe index is not a DatetimeIndex, which is required for date operations.")
    st.stop()

# Convert to python dates properly - use a safer approach
min_date = datetime.strptime(str(full_df.index.min())[:10], '%Y-%m-%d').date()
max_date = datetime.strptime(str(full_df.index.max())[:10], '%Y-%m-%d').date()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

if start_date and end_date and start_date > end_date:
    st.error("Error: End date must fall after start date.")
else:
    filtered_df = full_df.loc[str(start_date):str(end_date)]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['stage_m'], mode='lines', name='Water Level'))
    fig_hist.update_layout(title="Historical Water Level Data", xaxis_title="Date", yaxis_title="Water Level (m)", height=500)
    st.plotly_chart(fig_hist, use_container_width=True)

# --- Model Selection ---
st.header("Analyze a Model")
available_models = [m for m in ['XGBoost', 'MLP', 'LSTM'] if models.get(m)]
model_selection = st.selectbox("Select the model to analyze:", available_models)

model = models.get(model_selection)
scaler = scalers.get(model_selection)

if not model or not scaler:
    st.error(f"{model_selection} model or scaler is not loaded. Please check the files in the 'models/' directory.")
    st.stop()

# --- Model-specific Parameters ---
N_PAST = 7  # History for LSTM/MLP
N_FUTURE = 7 # Days to predict

# --- Model Predictions Section ---
st.header("Model Predictions")
st.subheader("Forecast Water Level")
st.info(f"The {model_selection} model provides a {N_FUTURE}-day forecast with confidence intervals.")

# Add forecast start date selection - allow forecasting from today for real-time predictions
# Convert to python dates properly for date input widget
min_date_str = str(full_df.index.min())[:10]
dataset_end_str = str(full_df.index.max())[:10]
min_date = datetime.strptime(min_date_str, '%Y-%m-%d').date()
dataset_end_date = datetime.strptime(dataset_end_str, '%Y-%m-%d').date()

# For real-time forecasting, allow selection up to today (even if beyond dataset)
import datetime as dt
today_date = dt.date.today()
max_date = today_date  # Always allow forecasting from today

# Default to latest available data date, but allow forecasting up to today
default_forecast_date = min(dataset_end_date, today_date)

# Need N_PAST days of history, so earliest forecast date is min_date + N_PAST days
earliest_forecast_date = min_date + dt.timedelta(days=N_PAST)

col1, col2 = st.columns(2)
with col1:
    # Add confidence level selection
    confidence_level = st.selectbox(
        "Select confidence level:",
        options=[68, 95, 99],
        index=1,  # Default to 95%
        format_func=lambda x: f"{x}% Confidence Interval"
    )

with col2:
    # Add forecast start date selection
    forecast_start_date = st.date_input(
        "Select forecast start date:",
        value=default_forecast_date,
        min_value=earliest_forecast_date,
        max_value=max_date,
        help=f"Choose any date from {earliest_forecast_date} to {max_date}. Defaults to latest available data ({dataset_end_date})."
    )

# Convert selected date back to datetime for processing
forecast_start_datetime = pd.to_datetime(forecast_start_date)

if st.button("🚀 Generate Forecast", key=f"generate_{model_selection}"):
    with st.spinner("Generating forecast and calculating confidence intervals..."):
        
        # Calculate prediction errors for confidence intervals
        error_stats = calculate_prediction_errors(model_selection, model, scaler, full_df, N_PAST, N_FUTURE)
        
        # Handle real-time forecasting - use available data up to selected date or dataset end
        dataset_end_datetime = pd.Timestamp(str(full_df.index.max())[:19])
        
        if forecast_start_datetime <= dataset_end_datetime:
            # Forecast date is within dataset - use data up to selected date
            data_up_to_start = full_df[full_df.index <= forecast_start_datetime]
            actual_start_date = forecast_start_datetime
        else:
            # Forecast date is beyond dataset - use all available data
            data_up_to_start = full_df.copy()
            actual_start_date = dataset_end_datetime
            st.info(f"📊 **Real-time forecasting**: Selected date ({forecast_start_date}) is beyond dataset. Using most recent data ({actual_start_date.date()}) as starting point.")
        
        if len(data_up_to_start) < N_PAST:
            st.error(f"Not enough historical data. Need at least {N_PAST} days of data before the selected date.")
            st.stop()
        
        if model_selection == 'XGBoost':
            # Get the last N_PAST days before the forecast start date
            latest_data = data_up_to_start[scaler.feature_names_in_].tail(N_PAST)  # type: ignore
            scaled_data = scaler.transform(latest_data)
            X_pred = np.array([scaled_data])
            
            # Flatten the sequence for baseline XGBoost (same as MLP)
            X_pred_flat = X_pred.reshape(X_pred.shape[0], -1)
            
            prediction_scaled = model.predict(X_pred_flat)
            
            if prediction_scaled.ndim == 1:
                prediction_scaled = prediction_scaled.reshape(1, -1)
            
            n_features = len(scaler.feature_names_in_)
            temp_pred_array = np.zeros((prediction_scaled.shape[1], n_features))
            temp_pred_array[:, 0] = prediction_scaled[0]
            prediction_original = scaler.inverse_transform(temp_pred_array)[:, 0]

        elif model_selection == 'MLP':
            # Get the last N_PAST days before the forecast start date
            latest_data = data_up_to_start[scaler.feature_names_in_].tail(N_PAST)  # type: ignore
            scaled_data = scaler.transform(latest_data)
            X_pred = np.array([scaled_data])
            
            # Flatten the sequence for MLP (shape becomes (1, 49) for 7 timesteps * 7 features)
            X_pred_flat = X_pred.reshape(X_pred.shape[0], -1)
            
            prediction_scaled = model.predict(X_pred_flat)
            
            n_features = len(scaler.feature_names_in_)
            temp_pred_array = np.zeros((prediction_scaled.shape[1], n_features))
            temp_pred_array[:, 0] = prediction_scaled.flatten()
            prediction_original = scaler.inverse_transform(temp_pred_array)[:, 0]

        else: # LSTM
            # Get the last N_PAST days before the forecast start date
            latest_data = data_up_to_start[scaler.feature_names_in_].tail(N_PAST)  # type: ignore
            scaled_data = scaler.transform(latest_data)
            X_pred = np.array([scaled_data])
            
            prediction_scaled = model.predict(X_pred)
            
            n_features = len(scaler.feature_names_in_)
            temp_pred_array = np.zeros((prediction_scaled.shape[1], n_features))
            temp_pred_array[:, 0] = prediction_scaled.flatten()
            prediction_original = scaler.inverse_transform(temp_pred_array)[:, 0]

        # Generate forecast dates starting from the day after the actual start date
        forecast_dates = pd.to_datetime(actual_start_date) + pd.to_timedelta(np.arange(1, N_FUTURE + 1), 'D')
        
        # Calculate confidence intervals if error stats are available
        if error_stats:
            ci_key = f'ci_{confidence_level}'
            
            # Calculate upper and lower bounds for each forecast day
            lower_bounds = []
            upper_bounds = []
            
            for i in range(N_FUTURE):
                day_key = f'day_{i+1}'
                if day_key in error_stats:
                    error_margin = error_stats[day_key][ci_key]
                    lower_bounds.append(prediction_original[i] - error_margin)
                    upper_bounds.append(prediction_original[i] + error_margin)
                else:
                    # Fallback to no confidence interval
                    lower_bounds.append(prediction_original[i])
                    upper_bounds.append(prediction_original[i])
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates.strftime('%Y-%m-%d'), 
                'Forecasted Water Level (m)': prediction_original.round(2),
                f'Lower Bound ({confidence_level}% CI)': np.array(lower_bounds).round(2),
                f'Upper Bound ({confidence_level}% CI)': np.array(upper_bounds).round(2)
            })
        else:
            forecast_df = pd.DataFrame({
                'Date': forecast_dates.strftime('%Y-%m-%d'), 
                'Forecasted Water Level (m)': prediction_original.round(2)
            })
            lower_bounds = upper_bounds = None

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"{N_FUTURE}-Day Forecast:")
            st.write(f"Starting from: **{actual_start_date.date()}**")
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        with col2:
            fig_forecast = go.Figure()
            
            # Historical data - show context around the forecast start date
            context_days = 30
            context_start = max(0, len(data_up_to_start) - context_days)
            historical_context = data_up_to_start.iloc[context_start:]
            
            # Historical data
            fig_forecast.add_trace(go.Scatter(
                x=historical_context.index, 
                y=historical_context['stage_m'], 
                mode='lines+markers', 
                name='Historical', 
                line=dict(color='blue')
            ))
            
            # Highlight the forecast start point
            fig_forecast.add_trace(go.Scatter(
                x=[actual_start_date], 
                y=[data_up_to_start['stage_m'].iloc[-1]],  # type: ignore
                mode='markers', 
                name='Forecast Start', 
                marker=dict(color='green', size=10, symbol='star')
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_dates, 
                y=prediction_original, 
                mode='lines+markers', 
                name='Forecast', 
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            if error_stats and lower_bounds is not None and upper_bounds is not None:
                # Add confidence interval as filled area
                fig_forecast.add_trace(go.Scatter(
                    x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
                    y=np.concatenate([upper_bounds, lower_bounds[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name=f'{confidence_level}% Confidence Interval',
                    showlegend=True
                ))
            
            # Add actual values if they exist (for past forecasts)
            if forecast_dates[-1] <= full_df.index.max():
                actual_future = full_df[full_df.index.isin(forecast_dates)]
                if not actual_future.empty:
                    fig_forecast.add_trace(go.Scatter(
                        x=actual_future.index,
                        y=actual_future['stage_m'],
                        mode='lines+markers',
                        name='Actual (for comparison)',
                        line=dict(color='orange', width=3),
                        marker=dict(size=8)
                    ))
            
            fig_forecast.update_layout(
                title=f"Forecast from {actual_start_date.date()} - {N_FUTURE} Days Ahead with {confidence_level}% CI",
                xaxis_title="Date",
                yaxis_title="Water Level (m)"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
        # Show confidence interval interpretation
        if error_stats:
            st.info(f"""
            **Confidence Interval Interpretation:**
            
            The {confidence_level}% confidence interval means that, based on historical model performance, 
            we expect the actual water level to fall within the shaded area {confidence_level}% of the time.
            
            The intervals get wider for longer forecast horizons because prediction uncertainty increases over time.
            """)
            
        # Show forecast performance if actual data is available
        if forecast_dates[-1] <= full_df.index.max():
            st.success("💡 **Actual data is available for this forecast period!** You can compare the forecast accuracy above.")
        else:
            st.info("🔮 **This is a future forecast** - actual data is not yet available for comparison.")

# --- Model Performance Section ---
st.header("Model Performance")

tab1, tab2 = st.tabs(["Metrics", "Test Predictions vs Actual"])

with tab1:
    st.subheader("Metrics")
    
    if model_selection == 'XGBoost':
        # Use pre-computed metrics from training notebook
        st.write("**Day-by-Day Performance (Validation Set):**")
        
        metrics_data = [
            {'Day': 'Day 1', 'MAE': '0.1152 m', 'RMSE': '0.2362 m', 'R²': '0.9033'},
            {'Day': 'Day 2', 'MAE': '0.2333 m', 'RMSE': '0.4482 m', 'R²': '0.6514'},
            {'Day': 'Day 3', 'MAE': '0.3034 m', 'RMSE': '0.5538 m', 'R²': '0.4671'},
            {'Day': 'Day 4', 'MAE': '0.3377 m', 'RMSE': '0.5980 m', 'R²': '0.3780'},
            {'Day': 'Day 5', 'MAE': '0.3573 m', 'RMSE': '0.6201 m', 'R²': '0.3307'},
            {'Day': 'Day 6', 'MAE': '0.3716 m', 'RMSE': '0.6367 m', 'R²': '0.2940'},
            {'Day': 'Day 7', 'MAE': '0.3780 m', 'RMSE': '0.6427 m', 'R²': '0.2797'}
        ]
        
        # Display metrics in a nice table
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    else: # LSTM/MLP
        # Use pre-computed metrics from training notebooks
        st.write("**Day-by-Day Performance (Validation Set):**")
        
        if model_selection == 'MLP':
            metrics_data = [
                {'Day': 'Day 1', 'MAE': '0.1504 m', 'RMSE': '0.2873 m', 'R²': '0.8570'},
                {'Day': 'Day 2', 'MAE': '0.2422 m', 'RMSE': '0.4543 m', 'R²': '0.6417'},
                {'Day': 'Day 3', 'MAE': '0.3049 m', 'RMSE': '0.5549 m', 'R²': '0.4650'},
                {'Day': 'Day 4', 'MAE': '0.3377 m', 'RMSE': '0.5991 m', 'R²': '0.3756'},
                {'Day': 'Day 5', 'MAE': '0.3571 m', 'RMSE': '0.6250 m', 'R²': '0.3201'},
                {'Day': 'Day 6', 'MAE': '0.3814 m', 'RMSE': '0.6465 m', 'R²': '0.2722'},
                {'Day': 'Day 7', 'MAE': '0.3884 m', 'RMSE': '0.6550 m', 'R²': '0.2519'}
            ]
        else: # LSTM
            metrics_data = [
                {'Day': 'Day 1', 'MAE': '0.1326 m', 'RMSE': '0.2424 m', 'R²': '0.8982'},
                {'Day': 'Day 2', 'MAE': '0.2431 m', 'RMSE': '0.4457 m', 'R²': '0.6553'},
                {'Day': 'Day 3', 'MAE': '0.3086 m', 'RMSE': '0.5467 m', 'R²': '0.4807'},
                {'Day': 'Day 4', 'MAE': '0.3417 m', 'RMSE': '0.5930 m', 'R²': '0.3884'},
                {'Day': 'Day 5', 'MAE': '0.3606 m', 'RMSE': '0.6165 m', 'R²': '0.3385'},
                {'Day': 'Day 6', 'MAE': '0.3742 m', 'RMSE': '0.6312 m', 'R²': '0.3060'},
                {'Day': 'Day 7', 'MAE': '0.3839 m', 'RMSE': '0.6426 m', 'R²': '0.2798'}
            ]
        
        # Display metrics in a nice table
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Test set predictions vs actual values")
    st.info("This plot shows the model's 1-day ahead predictions against the actual values for the entire validation set.")
    
    fig_test = go.Figure()
    if model_selection == 'XGBoost':
        # For baseline XGBoost, use dynamic validation period up to today
        split_date_start = '2019-01-01'
        today = pd.Timestamp.now().normalize()
        dataset_end = pd.Timestamp(str(full_df.index.max())[:19])  # Convert to string first to avoid type issues
        validation_end = min(today, dataset_end)
        val_data = full_df[(full_df.index > split_date_start) & (full_df.index <= validation_end)]
        
        # Scale the validation data
        df_scaled_val = pd.DataFrame(scaler.transform(val_data[scaler.feature_names_in_]), 
                                    columns=scaler.feature_names_in_, 
                                    index=val_data.index)
        
        # Create sequences for the entire validation set
        X_val_xgb, y_val_xgb = create_sequences(df_scaled_val, N_PAST, N_FUTURE)
        
        if len(X_val_xgb) > 0:
            # Flatten sequences for XGBoost
            X_val_flat = X_val_xgb.reshape(X_val_xgb.shape[0], -1)
            y_pred_val_xgb_scaled = model.predict(X_val_flat)
            
            # Inverse transform predictions to original scale for plotting
            n_features = len(scaler.feature_names_in_)
            
            # Inverse transform predictions
            dummy_pred = np.zeros((len(y_pred_val_xgb_scaled.flatten()), n_features))
            dummy_pred[:, 0] = y_pred_val_xgb_scaled.flatten()
            y_pred_val_xgb_original = scaler.inverse_transform(dummy_pred)[:, 0].reshape(y_pred_val_xgb_scaled.shape)
            
            # Inverse transform actual values  
            dummy_actual = np.zeros((len(y_val_xgb.flatten()), n_features))
            dummy_actual[:, 0] = y_val_xgb.flatten()
            y_val_xgb_original = scaler.inverse_transform(dummy_actual)[:, 0].reshape(y_val_xgb.shape)
            
            # Create corresponding dates (accounting for the N_PAST offset)
            plot_dates = df_scaled_val.index[N_PAST:N_PAST+len(y_val_xgb)]
            
            # Extract 1-day ahead predictions (index 0 of the 7-day forecasts)
            actual_1day = y_val_xgb_original[:, 0]
            predicted_1day = y_pred_val_xgb_original[:, 0]
            
            fig_test.add_trace(go.Scatter(x=plot_dates, y=actual_1day, mode='lines', name='Actual', line=dict(color='blue')))
            fig_test.add_trace(go.Scatter(x=plot_dates, y=predicted_1day, mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
        else:
            st.warning("Not enough data to generate the comparison plot.")
    else: # LSTM/MLP
        # For LSTM/MLP, use dynamic validation period up to today
        split_date_start = '2019-01-01'
        today = pd.Timestamp.now().normalize()
        dataset_end = pd.Timestamp(str(full_df.index.max())[:19])  # Convert to string first to avoid type issues
        validation_end = min(today, dataset_end)
        val_data = full_df[(full_df.index > split_date_start) & (full_df.index <= validation_end)]
        
        # Scale the validation data
        df_scaled_val = pd.DataFrame(scaler.transform(val_data[scaler.feature_names_in_]), 
                                    columns=scaler.feature_names_in_, 
                                    index=val_data.index)
        
        # Create sequences for the entire validation set
        X_val_lstm, y_val_lstm = create_sequences(df_scaled_val, N_PAST, N_FUTURE)
        
        if len(X_val_lstm) > 0:
            # Get predictions for the entire validation set
            if model_selection == 'MLP':
                # Flatten sequences for MLP
                X_val_flat = X_val_lstm.reshape(X_val_lstm.shape[0], -1)
                y_pred_val_lstm_scaled = model.predict(X_val_flat)
            else: # LSTM
                y_pred_val_lstm_scaled = model.predict(X_val_lstm)
                # LSTM might return predictions with extra dimension, squeeze if needed
                if y_pred_val_lstm_scaled.ndim == 3:
                    y_pred_val_lstm_scaled = y_pred_val_lstm_scaled.squeeze(axis=-1)
            
            # Inverse transform predictions to original scale for plotting
            n_features = len(scaler.feature_names_in_)
            
            # Inverse transform predictions
            dummy_pred = np.zeros((len(y_pred_val_lstm_scaled.flatten()), n_features))
            dummy_pred[:, 0] = y_pred_val_lstm_scaled.flatten()
            y_pred_val_lstm_original = scaler.inverse_transform(dummy_pred)[:, 0].reshape(y_pred_val_lstm_scaled.shape)
            
            # Inverse transform actual values  
            dummy_actual = np.zeros((len(y_val_lstm.flatten()), n_features))
            dummy_actual[:, 0] = y_val_lstm.flatten()
            y_val_lstm_original = scaler.inverse_transform(dummy_actual)[:, 0].reshape(y_val_lstm.shape)
            
            # Create corresponding dates (accounting for the N_PAST offset)
            plot_dates = df_scaled_val.index[N_PAST:N_PAST+len(y_val_lstm)]
            
            # Extract 1-day ahead predictions (index 0 of the 7-day forecasts)
            actual_1day = y_val_lstm_original[:, 0]
            predicted_1day = y_pred_val_lstm_original[:, 0]
            
            fig_test.add_trace(go.Scatter(x=plot_dates, y=actual_1day, mode='lines', name='Actual', line=dict(color='blue', width=2)))
            fig_test.add_trace(go.Scatter(x=plot_dates, y=predicted_1day, mode='lines', name='Predicted', 
                                        line=dict(color='red', dash='dash', width=3), opacity=0.8))
        else:
            st.warning("Not enough data to generate the comparison plot.")

    fig_test.update_layout(title=f"{model_selection} Prediction vs Actual (Full Validation Set)", xaxis_title="Date", yaxis_title="Water Level (m)")
    st.plotly_chart(fig_test, use_container_width=True)