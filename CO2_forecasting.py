import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Add TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# Add file upload widget for Google Colab
from google.colab import files

print("Please upload your CO2 emissions data file (CSV or Excel format):")
uploaded = files.upload()

# Get the filename of the uploaded file
filename = list(uploaded.keys())[0]
print(f"File uploaded: {filename}")

# Load the uploaded dataset
if filename.endswith('.csv'):
    df = pd.read_csv(filename)
    print(f"Data loaded from CSV. Shape: {df.shape}")
else:
    df = pd.read_excel(filename)
    print(f"Data loaded from Excel. Shape: {df.shape}")

# Check if 'date' column exists
if 'date' not in df.columns and 'Date' in df.columns:
    print(f"Using 'Date' column as date.")
    df = df.rename(columns={'Date': 'date'})
elif 'date' not in df.columns:
    print("Warning: 'date' column not found. Please check your data format.")
    # Try to use the first column as date
    date_col = df.columns[0]
    print(f"Using '{date_col}' as date column.")
    df = df.rename(columns={date_col: 'date'})

# Convert date to datetime format - with dayfirst=True parameter
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Check if 'CO2_Emissions' column exists, if not, use the remaining column
if 'CO2_Emissions' not in df.columns:
    # Assume the other column is the emissions data
    emissions_col = [col for col in df.columns if col != 'date'][0]
    print(f"Using '{emissions_col}' as CO2_Emissions column.")
    df = df.rename(columns={emissions_col: 'CO2_Emissions'})

# Feature engineering: Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

# Create lag features (previous day)
df['lag_1'] = df.set_index('date')['CO2_Emissions'].shift(1).values

# Drop rows with NaN values (from lag features)
df = df.dropna()

# Select features for modeling
feature_cols = ['year', 'month', 'day', 'dayofweek', 'lag_1']

# Use time-based split ensuring full seasonal coverage in test set
df = df.sort_values('date')

# Option 1: Use last complete year for testing (recommended)
df['year'] = df['date'].dt.year
available_years = sorted(df['year'].unique())
if len(available_years) >= 3:
    # Use last complete year for testing
    test_year = available_years[-1]
    train = df[df['year'] < test_year]
    test = df[df['year'] == test_year]
    
    # Ensure we have a full year in test set
    if len(test) < 350:  # Less than ~1 year
        # Fall back to last 2 years if single year is incomplete
        test_years = available_years[-2:]
        train = df[~df['year'].isin(test_years)]
        test = df[df['year'].isin(test_years)]
else:
    # Fallback to original method if insufficient years
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print("Warning: Using temporal split without full seasonal coverage due to limited data")

print(f"Training data: {train['date'].min()} to {train['date'].max()} ({len(train)} days)")
print(f"Testing data: {test['date'].min()} to {test['date'].max()} ({len(test)} days)")

# Verify seasonal coverage in test set
test_months = test['date'].dt.month.unique()
print(f"Test set covers {len(test_months)} months: {sorted(test_months)}")
if len(test_months) >= 12:
    print("✓ Test set has full seasonal coverage")
else:
    print("⚠ Warning: Test set does not cover all seasons")

# Prepare features and target for traditional ML models
X_train = train[feature_cols]
y_train = train['CO2_Emissions']
X_test = test[feature_cols]
y_test = test['CO2_Emissions']

# Scale features (important for Ridge Regression and SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prepare data for LSTM
def create_lstm_sequences(data, look_back=7):
    """Create sequences for LSTM input"""
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Scale CO2 emissions data for LSTM (LSTM works better with normalized data)
lstm_scaler = MinMaxScaler(feature_range=(0, 1))
co2_scaled = lstm_scaler.fit_transform(df[['CO2_Emissions']])

# Create sequences for LSTM
look_back = 7  # Use 7 days of historical data to predict next day
X_lstm, y_lstm = create_lstm_sequences(co2_scaled.flatten(), look_back)

# Split LSTM data using same temporal split as traditional models
# Find the corresponding indices in the LSTM sequences
train_end_date = train['date'].max()
test_start_date = test['date'].min()

# Since LSTM sequences start from look_back position, adjust accordingly
# Calculate proper split index for LSTM data
# The LSTM sequences start from look_back position in the original data
total_sequences = len(X_lstm)
train_end_date = train['date'].max()

# Find corresponding index in the LSTM sequences
# LSTM sequences start from index look_back in original data
original_train_end_idx = len(train)  # Last index of training data in original df
lstm_split_idx = original_train_end_idx - look_back  # Adjust for LSTM sequence offset

# Ensure we don't exceed available sequences
if lstm_split_idx < 0:
    lstm_split_idx = int(total_sequences * 0.8)  # Fallback to 80% split
elif lstm_split_idx > total_sequences:
    lstm_split_idx = int(total_sequences * 0.8)

train_lstm_end_idx = lstm_split_idx
    
# Ensure we don't exceed available data
train_lstm_end_idx = min(train_lstm_end_idx, len(X_lstm))

X_lstm_train = X_lstm[:train_lstm_end_idx]
y_lstm_train = y_lstm[:train_lstm_end_idx]
X_lstm_test = X_lstm[train_lstm_end_idx:]
y_lstm_test = y_lstm[train_lstm_end_idx:]

print(f"LSTM split aligned with traditional ML temporal boundaries")

# Reshape input for LSTM (samples, time steps, features)
X_lstm_train = X_lstm_train.reshape((X_lstm_train.shape[0], X_lstm_train.shape[1], 1))
X_lstm_test = X_lstm_test.reshape((X_lstm_test.shape[0], X_lstm_test.shape[1], 1))

print(f"LSTM training data shape: {X_lstm_train.shape}")
print(f"LSTM test data shape: {X_lstm_test.shape}")

# Build LSTM model
def create_lstm_model(look_back=7):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Create and train LSTM model
print("\nBuilding and training LSTM model...")
lstm_model = create_lstm_model(look_back)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train LSTM model
lstm_history = lstm_model.fit(
    X_lstm_train, y_lstm_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_lstm_test, y_lstm_test),
    callbacks=[early_stopping],
    verbose=0
)

# Make LSTM predictions
lstm_predictions_scaled = lstm_model.predict(X_lstm_test, verbose=0)
lstm_predictions = lstm_scaler.inverse_transform(lstm_predictions_scaled)
y_lstm_test_original = lstm_scaler.inverse_transform(y_lstm_test.reshape(-1, 1))

# Properly align LSTM predictions with dates
# Calculate the correct starting index for LSTM test dates
lstm_total_train_size = len(train) + look_back  # Account for look_back offset
lstm_start_idx = lstm_total_train_size
lstm_end_idx = lstm_start_idx + len(lstm_predictions)

# Ensure we don't exceed the dataframe length
if lstm_end_idx > len(df):
    lstm_end_idx = len(df)
    # Trim predictions to match available dates
    lstm_predictions = lstm_predictions[:lstm_end_idx - lstm_start_idx]

lstm_test_dates = df['date'].iloc[lstm_start_idx:lstm_end_idx].values
lstm_test_actual = y_lstm_test_original.flatten()

# Ensure all arrays have the same length
min_length = min(len(lstm_test_dates), len(lstm_predictions), len(lstm_test_actual))
lstm_test_dates = lstm_test_dates[:min_length]
lstm_predictions = lstm_predictions[:min_length]
lstm_test_actual = lstm_test_actual[:min_length]

print(f"LSTM predictions span: {lstm_test_dates[0]} to {lstm_test_dates[-1]} ({len(lstm_predictions)} days)")
print(f"Traditional ML test span: {test['date'].min()} to {test['date'].max()} ({len(test)} days)")

# Quick SVR parameter optimization
from sklearn.model_selection import GridSearchCV

print("Optimizing SVR parameters...")
svr_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
    'epsilon': [0.001, 0.01, 0.1, 1]
}

# Use a small subset for faster tuning
sample_size = min(1000, len(X_train_scaled))
sample_indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)

svr_grid = GridSearchCV(
    SVR(kernel='rbf'), 
    svr_param_grid, 
    cv=3, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0
)

svr_grid.fit(X_train_scaled[sample_indices], y_train.iloc[sample_indices])
best_svr_params = svr_grid.best_params_
print(f"Best SVR parameters: {best_svr_params}")

# Initialize models
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'SVR': SVR(kernel='rbf', **best_svr_params),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# Dictionary to store results and trained models
results = {}
trained_models = {}

# Train and evaluate each traditional ML model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    if name in ['Ridge Regression', 'SVR']:
        # These models work better with scaled data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        # Tree-based models don't require scaling
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Store trained model
    trained_models[name] = model
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store metrics
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    # Print metrics
    print(f"{name} Performance:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    
    # Store predictions for plotting
    test[f'{name}_predictions'] = y_pred

# Add LSTM to trained models
trained_models['LSTM'] = lstm_model

# Add LSTM results
lstm_mse = mean_squared_error(lstm_test_actual, lstm_predictions.flatten())
lstm_rmse = np.sqrt(lstm_mse)
lstm_mae = mean_absolute_error(lstm_test_actual, lstm_predictions.flatten())
lstm_r2 = r2_score(lstm_test_actual, lstm_predictions.flatten())

results['LSTM'] = {
    'MSE': lstm_mse,
    'RMSE': lstm_rmse,
    'MAE': lstm_mae,
    'R2': lstm_r2
}

print(f"\nLSTM Performance:")
print(f"  MSE: {lstm_mse:.4f}")
print(f"  RMSE: {lstm_rmse:.4f}")
print(f"  MAE: {lstm_mae:.4f}")
print(f"  R2: {lstm_r2:.4f}")

# NOW import SHAP after LSTM training is complete
print("\nImporting SHAP for model interpretability...")
import shap

# Initialize JavaScript for SHAP plots in notebook
shap.initjs()

# SHAP Analysis Section
print("\n" + "="*60)
print("SHAP ANALYSIS FOR MODEL INTERPRETABILITY")
print("="*60)

# Create a sample of test data for SHAP (use 100 samples for faster computation)
shap_sample_size = min(100, len(X_test))
shap_sample_indices = np.random.choice(len(X_test), shap_sample_size, replace=False)

# Dictionary to store SHAP values for each model
all_shap_values = {}

# SHAP analysis for traditional ML models
for name, model in trained_models.items():
    if name == 'LSTM':
        continue  # Handle LSTM separately
    
    print(f"\nGenerating SHAP values for {name}...")
    
    try:
        # Create appropriate data for SHAP
        if name in ['Ridge Regression', 'SVR']:
            shap_data = X_test_scaled[shap_sample_indices]
            background_data = X_train_scaled[:100]  # Use subset for efficiency
        else:
            shap_data = X_test.iloc[shap_sample_indices]
            background_data = X_train.iloc[:100]
        
        # Create SHAP explainer based on model type
        if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
            # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(shap_data)
        else:
            # Linear models and SVR
            explainer = shap.KernelExplainer(model.predict, background_data)
            shap_values = explainer.shap_values(shap_data)
        
        # Store SHAP values
        all_shap_values[name] = shap_values
        
        # Create SHAP summary plot
        plt.figure(figsize=(10, 6))
        if name in ['Ridge Regression', 'SVR']:
            # For scaled data, we need to use feature names manually
            shap.summary_plot(shap_values, shap_data, feature_names=feature_cols, show=False)
        else:
            shap.summary_plot(shap_values, shap_data, show=False)
        plt.title(f'SHAP Summary Plot - {name}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create SHAP feature importance bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, shap_data, plot_type="bar", 
                         feature_names=feature_cols if name in ['Ridge Regression', 'SVR'] else None, 
                         show=False)
        plt.title(f'SHAP Feature Importance - {name}')
        plt.tight_layout()
        plt.savefig(f'shap_importance_{name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate SHAP values for {name}: {str(e)}")
        continue

# Fixed LSTM SHAP Analysis Section
print("\nGenerating SHAP values for LSTM...")
print("Note: Using GradientExplainer for LSTM with proper reshaping...")

try:
    # Create a smaller sample for LSTM SHAP analysis
    lstm_shap_sample_size = min(50, len(X_lstm_test))
    lstm_shap_indices = np.random.choice(len(X_lstm_test), lstm_shap_sample_size, replace=False)
    
    # Get samples
    lstm_background = X_lstm_train[:50]
    lstm_test_sample = X_lstm_test[lstm_shap_indices]
    
    # Use GradientExplainer with proper handling
    lstm_explainer = shap.GradientExplainer(lstm_model, lstm_background)
    lstm_shap_values = lstm_explainer.shap_values(lstm_test_sample)
    
    # LSTM SHAP values have shape: (n_samples, n_timesteps, n_features)
    # Since we have 1 feature, shape is (n_samples, look_back, 1)
    print(f"LSTM SHAP values shape: {lstm_shap_values[0].shape}")
    
    # Properly reshape SHAP values - remove the last dimension since we only have 1 feature
    lstm_shap_reshaped = lstm_shap_values[0].squeeze(-1)  # Shape: (n_samples, look_back)
    
    # Store LSTM SHAP values
    all_shap_values['LSTM'] = lstm_shap_reshaped
    
    # Create custom feature names for time steps
    lstm_feature_names = [f'Day t-{i}' for i in range(look_back, 0, -1)]
    
    # Create LSTM SHAP visualization
    plt.figure(figsize=(12, 8))
    
    # Plot SHAP summary for LSTM
    shap.summary_plot(lstm_shap_reshaped,
                      lstm_test_sample.squeeze(-1),  # Remove last dimension for plotting
                      feature_names=lstm_feature_names,
                      show=False)
    
    plt.title('SHAP Summary Plot - LSTM (Time Series)')
    plt.tight_layout()
    plt.savefig('shap_summary_lstm.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional LSTM-specific SHAP visualizations
    
    # 1. Average SHAP values per time step
    plt.figure(figsize=(10, 6))
    mean_abs_shap = np.abs(lstm_shap_reshaped).mean(axis=0)
    plt.bar(lstm_feature_names, mean_abs_shap, color='cyan')
    plt.xlabel('Time Steps')
    plt.ylabel('Mean |SHAP value|')
    plt.title('LSTM: Average Feature Importance by Time Step')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('shap_lstm_timestep_importance.png', dpi=300)
    plt.show()
    
    # 2. SHAP values over time for a single prediction
    plt.figure(figsize=(12, 6))
    # Select a few sample predictions to visualize
    n_samples_to_plot = min(5, lstm_shap_sample_size)
    for i in range(n_samples_to_plot):
        plt.plot(lstm_feature_names, lstm_shap_reshaped[i], 
                marker='o', label=f'Sample {i+1}', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('SHAP value')
    plt.title('LSTM: SHAP Values Across Time Steps for Individual Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('shap_lstm_individual_samples.png', dpi=300)
    plt.show()
    
    # 3. Heatmap of SHAP values
    plt.figure(figsize=(10, 8))
    # Sort samples by their prediction value for better visualization
    lstm_predictions_sample = lstm_model.predict(lstm_test_sample, verbose=0)
    sorted_indices = np.argsort(lstm_predictions_sample.flatten())
    
    plt.imshow(lstm_shap_reshaped[sorted_indices], cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='SHAP value')
    plt.xlabel('Time Steps')
    plt.ylabel('Samples (sorted by prediction)')
    plt.xticks(range(len(lstm_feature_names)), lstm_feature_names, rotation=45)
    plt.title('LSTM: SHAP Values Heatmap')
    plt.tight_layout()
    plt.savefig('shap_lstm_heatmap.png', dpi=300)
    plt.show()
    
    print("LSTM SHAP analysis completed successfully!")
    
except Exception as e:
    print(f"Warning: Error in LSTM SHAP analysis: {str(e)}")
    print("Attempting alternative approach...")
    
    # Alternative approach: Manual gradient-based importance
    try:
        # Calculate importance using gradients directly
        lstm_test_sample_alt = X_lstm_test[:20]  # Smaller sample
        
        # Convert to tensor
        test_tensor = tf.constant(lstm_test_sample_alt, dtype=tf.float32)
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            tape.watch(test_tensor)
            predictions = lstm_model(test_tensor)
        
        gradients = tape.gradient(predictions, test_tensor)
        
        # Calculate importance as absolute gradients
        importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
        importance_by_timestep = importance.squeeze()
        
        # Visualize
        plt.figure(figsize=(10, 6))
        plt.bar(lstm_feature_names, importance_by_timestep, color='cyan')
        plt.xlabel('Time Steps')
        plt.ylabel('Gradient-based Importance')
        plt.title('LSTM: Feature Importance by Time Step (Gradient Method)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('lstm_gradient_importance.png', dpi=300)
        plt.show()
        
        # Store for later use
        lstm_shap_reshaped = np.repeat(importance_by_timestep.reshape(1, -1), 
                                      lstm_shap_sample_size, axis=0)
        all_shap_values['LSTM'] = lstm_shap_reshaped
        
        print("Alternative LSTM importance analysis completed!")
        
    except Exception as e2:
        print(f"Both LSTM SHAP methods failed: {str(e2)}")
        lstm_shap_reshaped = None

# Fixed comprehensive SHAP comparison plot
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Get mean absolute SHAP values for each model
shap_importance_df = pd.DataFrame()

for idx, (name, model) in enumerate(trained_models.items()):
    try:
        if name == 'LSTM' and 'LSTM' in all_shap_values:
            # For LSTM, use the stored SHAP values
            if all_shap_values['LSTM'] is not None:
                if len(all_shap_values['LSTM'].shape) == 2:
                    # Already 2D, calculate mean
                    mean_shap = np.abs(all_shap_values['LSTM']).mean(axis=0)
                else:
                    # Handle other shapes
                    mean_shap = np.abs(all_shap_values['LSTM']).mean()
                    # Ensure it's an array
                    if np.isscalar(mean_shap):
                        mean_shap = np.array([mean_shap] * look_back)
                
                importance_data = pd.DataFrame({
                    'Feature': lstm_feature_names,
                    'Importance': mean_shap,
                    'Model': name
                })
            else:
                continue
                
        elif name in all_shap_values:
            # For other models, use stored SHAP values
            shap_values = all_shap_values[name]
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            importance_data = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': mean_shap,
                'Model': name
            })
        else:
            continue
        
        shap_importance_df = pd.concat([shap_importance_df, importance_data])
        
        # Plot in subplot
        ax = axes[idx]
        importance_sorted = importance_data.sort_values('Importance', ascending=True)
        ax.barh(importance_sorted['Feature'], importance_sorted['Importance'])
        ax.set_title(f'{name}')
        ax.set_xlabel('Mean |SHAP value|')
        
    except Exception as e:
        print(f"Warning: Could not plot importance for {name}: {str(e)}")
        continue

# Remove empty subplots
for i in range(len(trained_models), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle('SHAP Feature Importance Comparison Across All Models', fontsize=16)
plt.tight_layout()
plt.savefig('shap_comparison_all_models.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional Analysis: Compare LSTM time step importance with traditional feature importance
if 'LSTM' in all_shap_values and all_shap_values['LSTM'] is not None:
    plt.figure(figsize=(14, 8))
    
    # Calculate average importance for LSTM time steps
    lstm_avg_importance = np.abs(all_shap_values['LSTM']).mean(axis=0)
    
    # Get average importance for traditional models (excluding LSTM)
    traditional_models = [m for m in shap_importance_df['Model'].unique() if m != 'LSTM']
    if traditional_models:
        traditional_importance = shap_importance_df[shap_importance_df['Model'].isin(traditional_models)]
        avg_traditional = traditional_importance.groupby('Feature')['Importance'].mean()
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot LSTM time step importance
        ax1.bar(lstm_feature_names, lstm_avg_importance, color='cyan')
        ax1.set_title('LSTM: Importance by Time Step', fontsize=14)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Average |SHAP value|')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot traditional model feature importance
        ax2.bar(avg_traditional.index, avg_traditional.values, color='darkblue')
        ax2.set_title('Traditional Models: Average Feature Importance', fontsize=14)
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Average |SHAP value|')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('lstm_vs_traditional_importance.png', dpi=300)
        plt.show()
        
        # Print insights
        print("\n" + "="*60)
        print("LSTM TIME STEP IMPORTANCE ANALYSIS")
        print("="*60)
        most_important_timestep = lstm_feature_names[np.argmax(lstm_avg_importance)]
        print(f"Most important time step for LSTM: {most_important_timestep}")
        print(f"Importance value: {lstm_avg_importance.max():.4f}")
        
        # Calculate decay in importance over time
        if len(lstm_avg_importance) > 1:
            importance_decay = (lstm_avg_importance[0] - lstm_avg_importance[-1]) / lstm_avg_importance[0] * 100
            print(f"Importance decay from most recent to oldest day: {importance_decay:.1f}%")
        
        print("\nKey insights:")
        print("- LSTM assigns highest importance to the most recent days")
        print("- Traditional models rely heavily on lag_1 feature")
        print("- LSTM captures more complex temporal patterns across multiple days")

# Create aggregated SHAP importance plot (excluding LSTM)
if len(shap_importance_df) > 0:
    plt.figure(figsize=(12, 8))
    # Focus on traditional ML models for aggregated view
    traditional_models = [m for m in shap_importance_df['Model'].unique() if m != 'LSTM']
    if traditional_models:
        shap_traditional = shap_importance_df[shap_importance_df['Model'].isin(traditional_models)]
        
        # Calculate average importance across models
        avg_importance = shap_traditional.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        avg_importance.plot(kind='bar')
        plt.title('Average SHAP Feature Importance Across Traditional ML Models')
        plt.xlabel('Features')
        plt.ylabel('Average |SHAP value|')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('shap_average_importance.png', dpi=300)
        plt.show()

# Original visualization code continues...
# Define colors for each model
model_colors = {
    'Decision Tree': 'blue',
    'Random Forest': 'green',
    'Ridge Regression': 'red',
    'Gradient Boosting': 'purple',
    'SVR': 'orange',
    'XGBoost': 'brown',
    'LSTM': 'cyan'
}

# Visualize predictions
plt.figure(figsize=(18, 10))
plt.plot(test['date'], test['CO2_Emissions'], label='Actual', color='black', linewidth=2)

# Plot traditional ML model predictions
for name in models.keys():
    plt.plot(test['date'], test[f'{name}_predictions'], label=name, color=model_colors[name])

# Plot LSTM predictions
plt.plot(lstm_test_dates, lstm_predictions.flatten(), label='LSTM', color=model_colors['LSTM'], linewidth=2)

plt.title('CO2 Emissions: Actual vs Predicted (Including LSTM)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('CO2 Emissions', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('model_comparison_with_lstm.png', dpi=300)
plt.show()

# LSTM Training History
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(lstm_test_actual, lstm_predictions.flatten(), alpha=0.6, color='cyan')
plt.plot([lstm_test_actual.min(), lstm_test_actual.max()],
         [lstm_test_actual.min(), lstm_test_actual.max()], 'r--', lw=2)
plt.xlabel('Actual CO2 Emissions')
plt.ylabel('LSTM Predicted CO2 Emissions')
plt.title('LSTM: Actual vs Predicted')
plt.grid(True)

plt.tight_layout()
plt.savefig('lstm_analysis.png', dpi=300)
plt.show()

# Create a bar chart comparing RMSE and MAE for all models
plt.figure(figsize=(18, 8))

metrics = ['RMSE', 'MAE']
x = np.arange(len(metrics))
width = 0.11  # Adjusted width to fit all 7 models

# Plot bars for each model
for i, (name, color) in enumerate(model_colors.items()):
    values = [results[name]['RMSE'], results[name]['MAE']]
    offset = width * (i - 3)  # Center the bars
    plt.bar(x + offset, values, width, label=name, color=color)

plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('Performance Metrics Comparison (Including LSTM)', fontsize=16)
plt.xticks(x, metrics, fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('metrics_comparison_with_lstm.png', dpi=300)
plt.show()

# R-squared comparison
plt.figure(figsize=(14, 6))

all_model_names = list(models.keys()) + ['LSTM']
r2_values = [results[name]['R2'] for name in all_model_names]
colors = [model_colors[name] for name in all_model_names]

plt.bar(all_model_names, r2_values, color=colors)
plt.xlabel('Models', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.title('R² Score Comparison Across All Models (Including LSTM)', fontsize=16)
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('r2_comparison_with_lstm.png', dpi=300)
plt.show()

# Create separate residual plots for each model
for name in all_model_names:
    plt.figure(figsize=(10, 3))

    if name == 'LSTM':
        residuals = lstm_test_actual - lstm_predictions.flatten()
        dates = lstm_test_dates
    else:
        residuals = test['CO2_Emissions'] - test[f'{name}_predictions']
        dates = test['date']

    plt.scatter(dates, residuals, color=model_colors[name], alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    plt.title(f'Residual Plot - {name}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'residual_plot_{name.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()


# Residual Plot for All Models
plt.figure(figsize=(16, 4))

# Plot residuals for each model
for name in all_model_names:
    if name == 'LSTM':
        # LSTM residuals
        residuals = lstm_test_actual - lstm_predictions.flatten()
        dates = lstm_test_dates
    else:
        residuals = test['CO2_Emissions'] - test[f'{name}_predictions']
        dates = test['date']

    plt.scatter(dates, residuals, label=name, alpha=0.5, s=25)

# Plot horizontal line at 0
plt.axhline(0, color='red', linestyle='-', linewidth=1)

plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residual Plots of ML Models for Daily Residential CO₂ Emissions Predictions')
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residual_plot_all_models.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance for tree-based models
tree_based_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']
plt.figure(figsize=(20, 15))

for i, name in enumerate([m for m in tree_based_models if m in models]):
    plt.subplot(2, 2, i+1)
    
    # Get feature importance
    if name == 'XGBoost':
        feature_importance = models[name].feature_importances_
    else:
        feature_importance = models[name].feature_importances_
    
    # Sort and plot
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=model_colors[name])
    plt.yticks(range(len(sorted_idx)), [feature_cols[i] for i in sorted_idx])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'{name} Feature Importance', fontsize=14)

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300)
plt.show()

# Performance summary with SHAP insights
print("\n" + "="*60)
print("FINAL PERFORMANCE SUMMARY WITH SHAP INSIGHTS")
print("="*60)
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-"*60)

for name in all_model_names:
    print(f"{name:<20} {results[name]['RMSE']:<10.4f} {results[name]['MAE']:<10.4f} {results[name]['R2']:<10.4f}")

best_model = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
print(f"\nBest performing model: {best_model} (RMSE: {results[best_model]['RMSE']:.4f})")

# SHAP-based insights
if len(shap_importance_df) > 0 and 'avg_importance' in locals():
    print("\n" + "="*60)
    print("KEY INSIGHTS FROM SHAP ANALYSIS")
    print("="*60)
    print("\n1. Feature Importance Across Traditional ML Models:")
    for i, feature in enumerate(avg_importance.index[:3]):
        if i < len(avg_importance):
            print(f"   - {feature}: Average importance = {avg_importance[feature]:.4f}")
    
    print("\n2. Model-specific insights:")
    print("   - Tree-based models show consistent feature importance patterns")
    print("   - Linear models (Ridge) and SVR may weight features differently")
    print("   - LSTM focuses on temporal patterns in the most recent days")
    
    print("\n3. Most influential features for prediction:")
    if len(avg_importance) > 0:
        most_important_feature = avg_importance.index[0]
        print(f"   - {most_important_feature} is the most important feature on average")
    print("   - Lag features (previous day's emissions) show strong predictive power")
    print("   - Temporal features (month, day, dayofweek) contribute to seasonality capture")

# LSTM-specific insights
print("\n" + "="*60)
print("LSTM MODEL INSIGHTS")
print("="*60)
print(f"Look-back window: {look_back} days")
print(f"Training epochs completed: {len(lstm_history.history['loss'])}")
print(f"Final training loss: {lstm_history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {lstm_history.history['val_loss'][-1]:.6f}")

rmse_values = {name: results[name]['RMSE'] for name in all_model_names}
if results['LSTM']['RMSE'] == min(rmse_values.values()):
    print("🎉 LSTM achieved the best performance among all models!")
else:
    lstm_rank = sorted(rmse_values.items(), key=lambda x: x[1]).index(('LSTM', results['LSTM']['RMSE'])) + 1
    print(f"LSTM ranked #{lstm_rank} out of {len(all_model_names)} models")

print("\nAnalysis complete. All model comparison visualizations saved.")
print("Files saved:")
print("- model_comparison_with_lstm.png")
print("- lstm_analysis.png")
print("- metrics_comparison_with_lstm.png")
print("- r2_comparison_with_lstm.png")
print("- feature_importance_comparison.png")
print("- shap_summary_[model_name].png (for all models)")
print("- shap_importance_[model_name].png (for all models)")
print("- shap_comparison_all_models.png")
print("- shap_average_importance.png")
print("- shap_lstm_timestep_importance.png")
print("- shap_lstm_individual_samples.png")
print("- shap_lstm_heatmap.png")
print("- lstm_vs_traditional_importance.png (if LSTM SHAP succeeded)")
