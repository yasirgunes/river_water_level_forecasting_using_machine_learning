# River Level Forecasting Dashboard
# Advanced Water Level Prediction System

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras  # type: ignore
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Optional, Tuple, Dict, Any
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page Configuration
st.set_page_config(
    page_title="River Level Forecasting Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stTab [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        font-weight: bold;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Asset Loading Functions
@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    """Load all forecasting models and their scalers."""
    models = {"models": {}, "scalers": {}, "status": {}}
    
    # Model configurations
    model_configs = {
        "Multivariate LSTM": {
            "model_file": "best_multivariate_model.keras",
            "scaler_file": "multivariate_scaler.joblib",
            "key": "multi_lstm"
        },
        "Univariate LSTM": {
            "model_file": "best_univariate_model.keras", 
            "scaler_file": "multivariate_scaler.joblib",  # Using multivariate scaler
            "key": "uni_lstm"
        },
        "XGBoost": {
            "model_file": "best_xgboost_model.joblib",
            "scaler_file": "xgboost_scaler.joblib",
            "key": "xgboost"
        }
    }
    
    for name, config in model_configs.items():
        try:
            if name == "XGBoost":
                models["models"][config["key"]] = joblib.load(config["model_file"])
            else:
                models["models"][config["key"]] = keras.models.load_model(config["model_file"])
            
            models["scalers"][config["key"]] = joblib.load(config["scaler_file"])
            models["status"][config["key"]] = "✅ Loaded"
            
        except FileNotFoundError:
            models["status"][config["key"]] = "❌ File Not Found"
        except Exception as e:
            models["status"][config["key"]] = f"❌ Error: {str(e)[:30]}..."
    
    return models

@st.cache_data
def load_dataset() -> Optional[pd.DataFrame]:
    """Load and prepare the river level dataset."""
    try:
        df = pd.read_csv('combined_dataset.csv', index_col='datetime', parse_dates=True)
        if isinstance(df, pd.DataFrame):
            df = df[['stage_m'] + [col for col in df.columns if col != 'stage_m']]
            return df  # type: ignore
        return None
    except FileNotFoundError:
        return None
    except Exception:
        return None

# Data Processing Functions
def create_sequences(data: np.ndarray, n_steps: int = 7) -> np.ndarray:
    """Create time series sequences for model input."""
    X = []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i, :])
    return np.array(X)

@st.cache_data
def generate_model_predictions(_model: Any, _scaler: Any, _df: pd.DataFrame, 
                             model_type: str = "multi_lstm") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate predictions for validation period."""
    split_date = '2019-01-01'
    val_df = _df[_df.index > split_date].copy()
    
    n_steps = 7
    n_features = _df.shape[1]
    
    # Prepare data based on model type
    if model_type == "uni_lstm":
        target_data = _df[['stage_m']].values
        scaled_data = _scaler.transform(_df)
        scaled_target = scaled_data[:, 0].reshape(-1, 1)
        X_val = create_sequences(scaled_target, n_steps)
    elif model_type == "xgboost":
        # For XGBoost, we need to prepare tabular data differently
        scaled_data = _scaler.transform(_df)
        X_sequences = create_sequences(scaled_data, n_steps)
        # Flatten sequences for XGBoost
        X_val = X_sequences.reshape(X_sequences.shape[0], -1)
    else:  # multivariate LSTM
        scaled_data = _scaler.transform(_df)
        X_val = create_sequences(scaled_data, n_steps)
    
    # Generate predictions
    if model_type == "xgboost":
        predictions_scaled = _model.predict(X_val)
        # Reshape for XGBoost output (should be 7 predictions per sample)
        predictions_scaled = predictions_scaled.reshape(-1, 7)
    else:
        predictions_scaled = _model.predict(X_val)
    
    # Inverse scale predictions
    total_predictions = predictions_scaled.size
    dummy_preds = np.zeros((total_predictions, n_features))
    dummy_preds[:, 0] = predictions_scaled.flatten()
    
    rescaled_preds = _scaler.inverse_transform(dummy_preds)
    
    if predictions_scaled.ndim > 1:
        predictions_original = rescaled_preds[:, 0].reshape(predictions_scaled.shape[0], predictions_scaled.shape[1])
    else:
        predictions_original = rescaled_preds[:, 0]
    
    # Create DataFrame
    pred_index = _df.index[n_steps:]
    columns = [f'Day {i+1}' for i in range(7)]
    
    preds_df = pd.DataFrame(predictions_original, index=pred_index, columns=columns)  # type: ignore
    preds_df = preds_df[preds_df.index > split_date]
    val_data_actuals = val_df[val_df.index.isin(preds_df.index)]
    
    return val_data_actuals, preds_df  # type: ignore

def calculate_comprehensive_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive performance metrics for each forecast day."""
    metrics = []
    for day in range(y_pred.shape[1]):
        day_col = f'Day {day+1}'
        true_shifted = y_true['stage_m'].shift(-day)
        common_index = y_pred.index.intersection(true_shifted.index)
        
        if common_index.empty:
            continue
            
        pred_vals = y_pred.loc[common_index, day_col]
        true_vals = true_shifted.loc[common_index]
        
        # Remove NaN values
        mask = ~(pd.isna(pred_vals) | pd.isna(true_vals))
        pred_vals_clean = pred_vals[mask]
        true_vals_clean = true_vals[mask]
        
        if len(pred_vals_clean) == 0:
            continue
        
        # Calculate metrics
        r2 = r2_score(true_vals_clean, pred_vals_clean)
        mse = mean_squared_error(true_vals_clean, pred_vals_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_vals_clean, pred_vals_clean)
        
        metrics.append({
            "Forecast Day": day + 1,
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MSE": mse
        })
    
    return pd.DataFrame(metrics).set_index("Forecast Day")

# Main Application
def main():
    # Header
    st.title("🌊 French Broad River: Advanced Water Level Forecasting System")
    st.markdown("""
    **Professional-grade water level prediction dashboard** featuring multiple machine learning models 
    for accurate 7-day forecasting. Built for hydrology research and operational water management.
    """)
    
    # Load assets
    with st.spinner("🔄 Loading models and data..."):
        assets = load_all_models()
        df = load_dataset()
    
    # Check data availability
    if df is None:
        st.error("❌ **Dataset not found!** Please ensure `combined_dataset.csv` is in the application directory.")
        st.stop()
    
    # Sidebar Status Panel
    with st.sidebar:
        st.header("🔧 System Status")
        
        model_names = {
            "multi_lstm": "Multivariate LSTM",
            "uni_lstm": "Univariate LSTM", 
            "xgboost": "XGBoost"
        }
        
        for key, name in model_names.items():
            status = assets["status"].get(key, "❌ Not Available")
            if "✅" in status:
                st.markdown(f"**{name}**: <span class='status-success'>{status}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**{name}**: <span class='status-error'>{status}</span>", unsafe_allow_html=True)
        
        # Dataset info
        st.markdown("---")
        st.markdown(f"**Dataset**: ✅ Loaded ({len(df):,} records)")
        # Type ignore for date handling - linter false positive
        min_date_str = pd.to_datetime(df.index.min()).strftime('%Y-%m-%d') if pd.notnull(df.index.min()) else "Unknown"  # type: ignore
        max_date_str = pd.to_datetime(df.index.max()).strftime('%Y-%m-%d') if pd.notnull(df.index.max()) else "Unknown"  # type: ignore
        st.markdown(f"**Date Range**: {min_date_str} to {max_date_str}")
        st.markdown(f"**Features**: {df.shape[1]} variables")
        
        # Quick stats
        st.markdown("---")
        st.subheader("📊 Quick Statistics")
        st.metric("Current Level", f"{df['stage_m'].iloc[-1]:.2f} m")
        st.metric("Average Level", f"{df['stage_m'].mean():.2f} m")
        st.metric("Max Recorded", f"{df['stage_m'].max():.2f} m")
        st.metric("Min Recorded", f"{df['stage_m'].min():.2f} m")
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Historical Data Explorer",
        "🎯 Model Performance Comparison", 
        "🔮 Interactive Forecasting",
        "📋 Technical Documentation"
    ])
    
    # Tab 1: Historical Data Explorer
    with tab1:
        st.header("📈 Historical River Water Level Analysis")
        st.markdown("""
        Explore the complete historical dataset of French Broad River water levels. 
        This interactive visualization allows you to examine patterns, trends, and seasonal variations.
        """)
        
        # Date range selector
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            start_date = st.date_input("📅 Start Date", df.index.min().date())
        with col2:
            end_date = st.date_input("📅 End Date", df.index.max().date())
        with col3:
            show_stats = st.checkbox("Show Statistics", value=True)
        
        # Filter data
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        filtered_df = df[mask]
        
        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df.index,
            y=filtered_df['stage_m'],
            mode='lines',
            name='Water Level',
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='<b>Date</b>: %{x}<br><b>Water Level</b>: %{y:.3f} m<extra></extra>'
        ))
        
        fig.update_layout(
            title="French Broad River Water Levels Over Time",
            xaxis_title="Date",
            yaxis_title="Water Level (meters)",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics panel
        if show_stats and not filtered_df.empty:
            st.subheader("📊 Period Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Mean Level", f"{filtered_df['stage_m'].mean():.3f} m")
            with col2:
                st.metric("Median Level", f"{filtered_df['stage_m'].median():.3f} m")
            with col3:
                st.metric("Std Deviation", f"{filtered_df['stage_m'].std():.3f} m")
            with col4:
                st.metric("Maximum", f"{filtered_df['stage_m'].max():.3f} m")
            with col5:
                st.metric("Minimum", f"{filtered_df['stage_m'].min():.3f} m")
        
        # Raw data viewer
        with st.expander("🔍 View Raw Dataset"):
            st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Tab 2: Model Performance Comparison
    with tab2:
        st.header("🎯 Advanced Model Performance Analysis")
        st.markdown("""
        Compare the performance of different machine learning models on unseen validation data (2019-2025).
        Each model predicts 7 days of water levels using the previous 7 days as input.
        """)
        
        # Model selector
        available_models = {}
        for key, status in assets["status"].items():
            if "✅" in status:
                model_names = {
                    "multi_lstm": "🧠 Multivariate LSTM",
                    "uni_lstm": "📊 Univariate LSTM",
                    "xgboost": "🌳 XGBoost Regressor"
                }
                available_models[model_names[key]] = key
        
        if not available_models:
            st.error("❌ No models are available for comparison. Please check the model files.")
            return
        
        selected_model_name = st.selectbox(
            "🔧 Select Model for Analysis:",
            options=list(available_models.keys()),
            index=0
        )
        model_key = available_models[selected_model_name]
        
        # Generate predictions
        with st.spinner(f"🔄 Generating predictions for {selected_model_name}..."):
            try:
                actuals, predictions = generate_model_predictions(
                    assets["models"][model_key], 
                    assets["scalers"][model_key], 
                    df, 
                    model_key
                )
                
                # Calculate metrics
                metrics_df = calculate_comprehensive_metrics(actuals, predictions)
                
                # Display metrics
                st.subheader(f"📈 Performance Metrics: {selected_model_name}")
                
                # Key metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_r2 = metrics_df['R²'].mean()
                    st.metric("Average R²", f"{avg_r2:.3f}", help="Coefficient of determination (higher is better)")
                with col2:
                    avg_rmse = metrics_df['RMSE'].mean()
                    st.metric("Average RMSE", f"{avg_rmse:.3f} m", help="Root Mean Square Error (lower is better)")
                with col3:
                    avg_mae = metrics_df['MAE'].mean()
                    st.metric("Average MAE", f"{avg_mae:.3f} m", help="Mean Absolute Error (lower is better)")
                with col4:
                    day1_r2 = metrics_df.loc[1, 'R²'] if 1 in metrics_df.index else 0
                    st.metric("Day 1 R²", f"{day1_r2:.3f}", help="Next-day prediction accuracy")
                
                # Detailed metrics table
                st.subheader("📋 Detailed Performance by Forecast Day")
                st.dataframe(
                    metrics_df.style.format({
                        'R²': '{:.3f}',
                        'RMSE': '{:.3f}',
                        'MAE': '{:.3f}', 
                        'MSE': '{:.3f}'
                    }).highlight_max(axis=0, subset=['R²'], color='lightgreen')
                    .highlight_min(axis=0, subset=['RMSE', 'MAE', 'MSE'], color='lightblue'),
                    use_container_width=True
                )
                
                # Visualization
                st.subheader("📊 Predictions vs. Actual Values (Validation Set)")
                
                # Sample data for visualization (to avoid overcrowding)
                sample_size = min(1000, len(predictions))
                sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
                sample_predictions = predictions.iloc[sample_indices].sort_index()
                sample_actuals = actuals[actuals.index.isin(sample_predictions.index)].sort_index()
                
                fig = go.Figure()
                
                # Actual values
                fig.add_trace(go.Scatter(
                    x=sample_actuals.index,
                    y=sample_actuals['stage_m'],
                    mode='lines',
                    name='Actual Values',
                    line=dict(color='blue', width=2),
                    opacity=0.8
                ))
                
                # Day 1 predictions
                fig.add_trace(go.Scatter(
                    x=sample_predictions.index,
                    y=sample_predictions['Day 1'],
                    mode='lines',
                    name='Day 1 Forecast',
                    line=dict(color='red', width=1.5, dash='dash'),
                    opacity=0.7
                ))
                
                # Day 7 predictions (if available)
                if 'Day 7' in sample_predictions.columns:
                    fig.add_trace(go.Scatter(
                        x=sample_predictions.index,
                        y=sample_predictions['Day 7'],
                        mode='lines',
                        name='Day 7 Forecast',
                        line=dict(color='orange', width=1.5, dash='dot'),
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    title=f"{selected_model_name}: Forecast vs. Actual Values (Sample of {sample_size} points)",
                    xaxis_title="Date",
                    yaxis_title="Water Level (meters)",
                    hovermode='x unified',
                    template="plotly_white",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model interpretation
                st.subheader("🧠 Model Performance Interpretation")
                
                interpretation = ""
                if avg_r2 > 0.8:
                    interpretation += "🟢 **Excellent** overall performance with strong predictive capability.\n"
                elif avg_r2 > 0.6:
                    interpretation += "🟡 **Good** performance with reliable short-term predictions.\n"
                else:
                    interpretation += "🔴 **Fair** performance - consider model improvements.\n"
                
                if day1_r2 > 0.85:
                    interpretation += "🎯 **Outstanding** next-day prediction accuracy.\n"
                elif day1_r2 > 0.7:
                    interpretation += "✅ **Reliable** next-day forecasting capability.\n"
                
                st.markdown(interpretation)
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                st.info("Please check that the model files are properly formatted and accessible.")
    
    # Tab 3: Interactive Forecasting
    with tab3:
        st.header("🔮 Interactive Water Level Forecasting")
        st.markdown("""
        Generate real-time 7-day water level forecasts for any date in the dataset. 
        Compare predictions with actual values when available.
        """)
        
        # Check for available models
        available_models = {k: v for k, v in assets["status"].items() if "✅" in v}
        if not available_models:
            st.error("❌ No models available for forecasting.")
            return
        
        # Layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎛️ Forecast Configuration")
            
            # Model selection for forecasting
            forecast_models = {}
            for key in available_models.keys():
                model_names = {
                    "multi_lstm": "🧠 Multivariate LSTM (Recommended)",
                    "uni_lstm": "📊 Univariate LSTM",
                    "xgboost": "🌳 XGBoost"
                }
                forecast_models[model_names[key]] = key
            
            selected_forecast_model = st.selectbox(
                "Select Forecasting Model:",
                options=list(forecast_models.keys()),
                index=0
            )
            forecast_model_key = forecast_models[selected_forecast_model]
            
            # Date selection
            min_date = df.index.min() + pd.DateOffset(days=7)
            max_date = df.index.max()
            
            selected_date = st.date_input(
                "📅 Forecast Start Date:",
                value=pd.to_datetime('2024-01-15').date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
                help="Model will use 7 days prior to this date for prediction"
            )
            
            # Input data display
            st.markdown("---")
            st.subheader("📋 Model Input Data")
            input_end = pd.to_datetime(selected_date) - pd.DateOffset(days=1)
            input_start = input_end - pd.DateOffset(days=6)
            input_df = df.loc[input_start:input_end]
            
            st.dataframe(
                input_df[['stage_m']].style.format({'stage_m': '{:.3f}'}),
                use_container_width=True
            )
            
            # Generate forecast button
            generate_forecast = st.button(
                "🚀 Generate 7-Day Forecast", 
                type="primary", 
                use_container_width=True
            )
        
        with col2:
            if generate_forecast:
                if len(input_df) == 7:
                    with st.spinner("🔄 Generating comprehensive forecast..."):
                        try:
                            # Get model and scaler
                            model = assets['models'][forecast_model_key]
                            scaler = assets['scalers'][forecast_model_key]
                            
                            # Prepare input data
                            input_scaled = scaler.transform(input_df)
                            
                            if forecast_model_key == "xgboost":
                                # XGBoost expects flattened input
                                input_reshaped = input_scaled.flatten().reshape(1, -1)
                                pred_scaled = model.predict(input_reshaped)
                                pred_scaled = pred_scaled.reshape(1, 7)
                            elif forecast_model_key == "uni_lstm":
                                # Univariate LSTM uses only water level
                                input_uni = input_scaled[:, 0].reshape(1, 7, 1)
                                pred_scaled = model.predict(input_uni)
                            else:
                                # Multivariate LSTM
                                input_reshaped = input_scaled.reshape(1, 7, input_df.shape[1])
                                pred_scaled = model.predict(input_reshaped)
                            
                            # Inverse transform predictions
                            dummy = np.zeros((pred_scaled.shape[1], df.shape[1]))
                            dummy[:, 0] = pred_scaled.flatten()
                            pred_unscaled = scaler.inverse_transform(dummy)[:, 0]
                            
                            # Create forecast DataFrame
                            forecast_dates = pd.date_range(start=selected_date, periods=7)
                            forecast_df = pd.DataFrame({
                                'Forecast': pred_unscaled
                            }, index=forecast_dates)
                            
                            # Get actual values for comparison
                            actual_end = pd.to_datetime(selected_date) + pd.DateOffset(days=6)
                            try:
                                actual_df = df.loc[pd.to_datetime(selected_date):actual_end, ['stage_m']]
                                actual_df.rename(columns={'stage_m': 'Actual'}, inplace=True)
                            except:
                                actual_df = pd.DataFrame({'Actual': [np.nan]*7}, index=forecast_dates)
                            
                            # Combine results
                            comparison_df = forecast_df.join(actual_df)
                            comparison_df['Error'] = comparison_df['Actual'] - comparison_df['Forecast']
                            comparison_df['Error %'] = (comparison_df['Error'] / comparison_df['Actual'] * 100)
                            
                            # Display results
                            st.subheader("📊 7-Day Forecast Results")
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Avg Forecast", f"{comparison_df['Forecast'].mean():.3f} m")
                            with col2:
                                st.metric("Max Forecast", f"{comparison_df['Forecast'].max():.3f} m")
                            with col3:
                                st.metric("Min Forecast", f"{comparison_df['Forecast'].min():.3f} m")
                            with col4:
                                trend = "📈" if comparison_df['Forecast'].iloc[-1] > comparison_df['Forecast'].iloc[0] else "📉"
                                st.metric("Trend", trend)
                            
                            # Detailed table
                            st.dataframe(
                                comparison_df.style.format({
                                    'Forecast': '{:.3f}',
                                    'Actual': '{:.3f}',
                                    'Error': '{:.3f}',
                                    'Error %': '{:.1f}%'
                                }),
                                use_container_width=True
                            )
                            
                            # Performance metrics (if actual data available)
                            valid_comparison = comparison_df.dropna()
                            if len(valid_comparison) > 0:
                                st.subheader("🎯 Forecast Accuracy Metrics")
                                
                                r2 = r2_score(valid_comparison['Actual'], valid_comparison['Forecast'])
                                rmse = np.sqrt(mean_squared_error(valid_comparison['Actual'], valid_comparison['Forecast']))
                                mae = mean_absolute_error(valid_comparison['Actual'], valid_comparison['Forecast'])
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R² Score", f"{r2:.3f}")
                                with col2:
                                    st.metric("RMSE", f"{rmse:.3f} m")
                                with col3:
                                    st.metric("MAE", f"{mae:.3f} m")
                            
                            # Visualization
                            st.subheader("📈 Forecast Visualization")
                            
                            # Historical context (last 30 days)
                            context_start = pd.to_datetime(selected_date) - pd.DateOffset(days=30)
                            context_data = df.loc[context_start:pd.to_datetime(selected_date)-pd.DateOffset(days=1)]
                            
                            fig = go.Figure()
                            
                            # Historical data
                            fig.add_trace(go.Scatter(
                                x=context_data.index,
                                y=context_data['stage_m'],
                                mode='lines',
                                name='Historical (30 days)',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Forecast
                            fig.add_trace(go.Scatter(
                                x=comparison_df.index,
                                y=comparison_df['Forecast'],
                                mode='lines+markers',
                                name='7-Day Forecast',
                                line=dict(color='red', width=3),
                                marker=dict(size=8)
                            ))
                            
                            # Actual values (if available)
                            if not valid_comparison.empty:
                                fig.add_trace(go.Scatter(
                                    x=valid_comparison.index,
                                    y=valid_comparison['Actual'],
                                    mode='markers',
                                    name='Actual Values',
                                    marker=dict(color='green', size=10, symbol='diamond')
                                ))
                            
                            # Forecast start line
                            fig.add_vline(
                                x=pd.to_datetime(selected_date),
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Forecast Start"
                            )
                            
                            fig.update_layout(
                                title=f"Water Level Forecast from {selected_date}",
                                xaxis_title="Date",
                                yaxis_title="Water Level (meters)",
                                hovermode='x unified',
                                template="plotly_white",
                                height=500,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Forecast generation failed: {str(e)}")
                else:
                    st.error("❌ Insufficient historical data for the selected date.")
            else:
                st.info("👆 Configure your forecast settings and click the button to generate predictions.")
    
    # Tab 4: Technical Documentation
    with tab4:
        st.header("📋 Technical Documentation")
        
        doc_tabs = st.tabs(["🏗️ Model Architecture", "📊 Dataset Information", "🔧 API Reference"])
        
        with doc_tabs[0]:
            st.subheader("🏗️ Model Architecture Overview")
            
            st.markdown("""
            ### 🧠 Multivariate LSTM
            - **Input**: 7-day sequences of multiple meteorological variables
            - **Architecture**: Multi-layer LSTM with dropout regularization
            - **Output**: 7-day water level predictions
            - **Advantages**: Captures complex temporal patterns and variable interactions
            
            ### 📊 Univariate LSTM  
            - **Input**: 7-day sequences of water level only
            - **Architecture**: Simplified LSTM focusing on temporal patterns
            - **Output**: 7-day water level predictions
            - **Advantages**: Robust performance with reduced complexity
            
            ### 🌳 XGBoost Regressor
            - **Input**: Flattened 7-day feature windows
            - **Architecture**: Gradient boosted trees with advanced regularization
            - **Output**: 7-day water level predictions
            - **Advantages**: Fast inference and excellent tabular data performance
            """)
            
            # Performance comparison
            st.subheader("⚡ Performance Characteristics")
            
            perf_data = {
                "Model": ["Multivariate LSTM", "Univariate LSTM", "XGBoost"],
                "Inference Speed": ["Medium", "Fast", "Very Fast"],
                "Memory Usage": ["High", "Medium", "Low"],
                "Day 1 Accuracy": ["Excellent", "Very Good", "Good"],
                "Day 7 Accuracy": ["Good", "Fair", "Fair"]
            }
            
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
        
        with doc_tabs[1]:
            st.subheader("📊 Dataset Information")
            
            if df is not None:
                st.markdown(f"""
                ### 📈 Dataset Overview
                - **Total Records**: {len(df):,} observations
                - **Date Range**: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
                - **Features**: {df.shape[1]} variables
                - **Frequency**: Hourly measurements
                - **Training Period**: Up to 2019-01-01
                - **Validation Period**: 2019-01-01 onwards
                """)
                
                st.subheader("📋 Feature Descriptions")
                feature_info = {
                    "Feature": df.columns.tolist(),
                    "Description": ["Target variable: River water stage in meters"] + 
                                 [f"Meteorological feature {i}" for i in range(len(df.columns)-1)],
                    "Unit": ["meters"] + ["various"] * (len(df.columns)-1)
                }
                
                st.dataframe(pd.DataFrame(feature_info), use_container_width=True)
                
                st.subheader("📊 Data Quality Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100
                    st.metric("Missing Data", f"{missing_pct:.2f}%")
                
                with col2:
                    duplicates = df.index.duplicated().sum()
                    st.metric("Duplicate Timestamps", f"{duplicates}")
                
                with col3:
                    outliers = len(df[np.abs(df['stage_m'] - df['stage_m'].mean()) > 3 * df['stage_m'].std()])
                    st.metric("Potential Outliers", f"{outliers}")
        
        with doc_tabs[2]:
            st.subheader("🔧 API Reference")
            
            st.markdown("""
            ### 🎯 Model Prediction Interface
            
            ```python
            # Load model and scaler
            model = keras.models.load_model('best_multivariate_model.keras')
            scaler = joblib.load('multivariate_scaler.joblib')
            
            # Prepare input data (7 days × features)
            input_scaled = scaler.transform(input_data)
            input_reshaped = input_scaled.reshape(1, 7, n_features)
            
            # Generate 7-day forecast
            predictions = model.predict(input_reshaped)
            
            # Inverse transform to original scale
            forecast = scaler.inverse_transform(predictions)[0]
            ```
            
            ### 📊 Data Requirements
            - **Input Shape**: (7, n_features) for multivariate models
            - **Scaling**: All features must be scaled using the provided scaler
            - **Temporal Order**: Data must be in chronological order
            - **Missing Values**: No missing values allowed in input sequence
            
            ### 🔄 Batch Processing
            For processing multiple forecasts efficiently:
            
            ```python
            # Batch prediction example
            batch_inputs = prepare_batch_sequences(data, window_size=7)
            batch_predictions = model.predict(batch_inputs)
            forecasts = inverse_transform_batch(batch_predictions, scaler)
            ```
            """)

if __name__ == "__main__":
    main()