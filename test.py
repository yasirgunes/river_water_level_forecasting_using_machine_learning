import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# 1. Load and Prepare Data
df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df_univariate = df[['stage_m']].copy()
df_univariate.interpolate(method='linear', inplace=True)

# 2. Split Data with a Gap (following paper methodology)
# Dataset: 1995-10-01 to 2025-06-01 (~29.67 years)
# Paper approach: 76% training, 23% validation with gap
# Training: 76% of 29.67 years ≈ 22.55 years → 1995-10-01 to 2018-04-21
# Gap: ~8 months (2018-04-21 to 2019-01-01)
# Validation: 2019-01-01 to 2025-06-01
split_date_train_end = '2018-04-21'
split_date_val_start = '2019-01-01'
train_data = df_univariate.loc[df_univariate.index < split_date_train_end]
val_data = df_univariate.loc[df_univariate.index >= split_date_val_start]

print(f"Training period: {train_data.index.min()} to {train_data.index.max()}")
print(f"Validation period: {val_data.index.min()} to {val_data.index.max()}")
print(f"Training samples: {len(train_data)} ({len(train_data)/(len(train_data)+len(val_data))*100:.1f}%)")
print(f"Validation samples: {len(val_data)} ({len(val_data)/(len(train_data)+len(val_data))*100:.1f}%)")
print(f"Gap between training and validation: {split_date_val_start} - {split_date_train_end}")

# 2.1. Visualize Dataset Split and Overview
plt.figure(figsize=(20, 12))

# Main time series plot with split visualization
plt.subplot(3, 2, (1, 2))
plt.plot(df_univariate.index, df_univariate['stage_m'], color='lightblue', alpha=0.7, linewidth=0.8, label='Full Dataset')
plt.plot(train_data.index, train_data['stage_m'], color='blue', alpha=0.8, linewidth=1, label=f'Training Data ({len(train_data)} samples)')
plt.plot(val_data.index, val_data['stage_m'], color='red', alpha=0.8, linewidth=1, label=f'Validation Data ({len(val_data)} samples)')

# Mark split points
train_end_date = pd.to_datetime(split_date_train_end)
val_start_date = pd.to_datetime(split_date_val_start)
plt.axvline(x=train_end_date, color='orange', linestyle='--', linewidth=2, label=f'Training End: {split_date_train_end}')
plt.axvline(x=val_start_date, color='green', linestyle='--', linewidth=2, label=f'Validation Start: {split_date_val_start}')

# Highlight gap period
plt.axvspan(train_end_date, val_start_date, alpha=0.3, color='yellow', label='Gap Period')

plt.title('Water Level Dataset with Training/Validation Split', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Water Level (m)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Training data statistics
plt.subplot(3, 2, 3)
plt.hist(train_data['stage_m'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Training Data Distribution', fontsize=14)
plt.xlabel('Water Level (m)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
stats_text = f'Mean: {train_data["stage_m"].mean():.2f}m\nStd: {train_data["stage_m"].std():.2f}m\nMin: {train_data["stage_m"].min():.2f}m\nMax: {train_data["stage_m"].max():.2f}m'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Validation data statistics
plt.subplot(3, 2, 4)
plt.hist(val_data['stage_m'], bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('Validation Data Distribution', fontsize=14)
plt.xlabel('Water Level (m)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
stats_text = f'Mean: {val_data["stage_m"].mean():.2f}m\nStd: {val_data["stage_m"].std():.2f}m\nMin: {val_data["stage_m"].min():.2f}m\nMax: {val_data["stage_m"].max():.2f}m'
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Yearly water level patterns
plt.subplot(3, 2, 5)
train_yearly = train_data.groupby(train_data.index.year)['stage_m'].mean()
val_yearly = val_data.groupby(val_data.index.year)['stage_m'].mean()
plt.plot(train_yearly.index, train_yearly.values, 'o-', color='blue', label='Training Years', linewidth=2, markersize=6)
plt.plot(val_yearly.index, val_yearly.values, 's-', color='red', label='Validation Years', linewidth=2, markersize=6)
plt.title('Average Annual Water Levels', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Average Water Level (m)')
plt.legend()
plt.grid(True, alpha=0.3)

# Monthly seasonality
plt.subplot(3, 2, 6)
train_monthly = train_data.groupby(train_data.index.month)['stage_m'].mean()
val_monthly = val_data.groupby(val_data.index.month)['stage_m'].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.plot(range(1, 13), train_monthly.values, 'o-', color='blue', label='Training', linewidth=2, markersize=6)
plt.plot(range(1, 13), val_monthly.values, 's-', color='red', label='Validation', linewidth=2, markersize=6)
plt.title('Seasonal Water Level Patterns', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Average Water Level (m)')
plt.xticks(range(1, 13), months, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display dataset summary
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"Total dataset span: {df_univariate.index.min()} to {df_univariate.index.max()}")
print(f"Total duration: {(df_univariate.index.max() - df_univariate.index.min()).days} days")
print(f"Training data: {len(train_data)} samples ({len(train_data)/(len(df_univariate))*100:.1f}%)")
print(f"Validation data: {len(val_data)} samples ({len(val_data)/(len(df_univariate))*100:.1f}%)")
gap_duration = (pd.to_datetime(split_date_val_start) - pd.to_datetime(split_date_train_end)).days
print(f"Gap duration: {gap_duration} days")
print("\nTraining Data Statistics:")
print(f"  Mean: {train_data['stage_m'].mean():.3f} m")
print(f"  Std:  {train_data['stage_m'].std():.3f} m")
print(f"  Min:  {train_data['stage_m'].min():.3f} m")
print(f"  Max:  {train_data['stage_m'].max():.3f} m")
print("\nValidation Data Statistics:")
print(f"  Mean: {val_data['stage_m'].mean():.3f} m")
print(f"  Std:  {val_data['stage_m'].std():.3f} m")
print(f"  Min:  {val_data['stage_m'].min():.3f} m")
print(f"  Max:  {val_data['stage_m'].max():.3f} m")
print("="*60)

# 3. Scale Data and Save Scaler with joblib
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_val_data = scaler.transform(val_data)

# Save the scaler using joblib
joblib.dump(scaler, 'univariate_scaler.joblib')

# 4. Create Sequences
N_PAST = 7   # Use 7 days of history
N_FUTURE = 7  # Predict 7 days into the future

def create_sequences(data, n_past, n_future):
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, 0])
        y.append(data[i:i + n_future, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(scaled_train_data, N_PAST, N_FUTURE)
X_val, y_val = create_sequences(scaled_val_data, N_PAST, N_FUTURE)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

print(f"New sequence shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# 5. Build Stacked LSTM Encoder-Decoder
n_features = X_train.shape[2]
UNITS = 128
DROPOUT_RATE = 0.15

# --- Encoder ---
inputs = Input(shape=(N_PAST, n_features))
# Layer 1
encoder_l1 = LSTM(UNITS, return_sequences=True, dropout=DROPOUT_RATE)(inputs)
# Layer 2
encoder_l2, state_h, state_c = LSTM(UNITS, return_sequences=False, dropout=DROPOUT_RATE, return_state=True)(encoder_l1)
encoder_states = [state_h, state_c]

# --- Decoder ---
decoder_inputs = RepeatVector(N_FUTURE)(encoder_l2)
# Layer 1
decoder_l1 = LSTM(UNITS, return_sequences=True, dropout=DROPOUT_RATE)(decoder_inputs, initial_state=encoder_states)
# Layer 2
decoder_l2 = LSTM(UNITS, return_sequences=True, dropout=DROPOUT_RATE)(decoder_l1)
# Output Layer
output = TimeDistributed(Dense(1))(decoder_l2)

model = Model(inputs=inputs, outputs=output)

# 6. Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.summary()

# 7. Train the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath='best_univariate_model.keras', save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# 8. Evaluation and Visualization
best_model = tf.keras.models.load_model('best_univariate_model.keras')
# Load the scaler using joblib
loaded_scaler = joblib.load('univariate_scaler.joblib')

predictions_scaled = best_model.predict(X_val).squeeze()
predictions_original = loaded_scaler.inverse_transform(predictions_scaled)
y_true_original = loaded_scaler.inverse_transform(y_val)

print("\n--- Model Performance on Validation Set ---")
for i in range(N_FUTURE):
    day = i + 1
    mae = mean_absolute_error(y_true_original[:, i], predictions_original[:, i])
    rmse = np.sqrt(mean_squared_error(y_true_original[:, i], predictions_original[:, i]))
    r2 = r2_score(y_true_original[:, i], predictions_original[:, i])
    print(f"Day {day} Ahead -> MAE: {mae:.4f} m, RMSE: {rmse:.4f} m, R²: {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss (MAE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MAE)')
plt.title('Model Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.grid(True)
plt.show()

val_dates = val_data.index[N_PAST:len(y_true_original) + N_PAST]
plt.figure(figsize=(15, 8))
plt.plot(val_dates, y_true_original[:, 0], label='Actual Values', color='blue', alpha=0.8)
plt.plot(val_dates, predictions_original[:, 0], label='Day 1 Forecast', color='red', linestyle='--', alpha=0.9)
plt.plot(val_dates, predictions_original[:, 2], label='Day 3 Forecast', color='orange', linestyle=':', alpha=0.8)
plt.plot(val_dates, predictions_original[:, 6], label='Day 7 Forecast', color='green', linestyle='-.', alpha=0.8)
plt.title('7-Day Water Level Forecast vs. Actual')
plt.xlabel('Date')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True)
plt.show()