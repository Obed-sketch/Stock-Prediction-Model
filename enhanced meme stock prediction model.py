# -*- coding: utf-8 -*-
"""
Enhanced Meme Stock Predictor with Cross-Validation, Backtesting, and Deployment
"""
# %% [1] Additional Imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import time
from flask import Flask, request, jsonify
import joblib
from datetime import date

# %% [2] Updated Model Building with Cross-Validation
def train_with_crossval(X, y, model, n_splits=5):
    """
    Time-series cross-validation for LSTM model
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    training_times = []
    memory_usage = []

    for train_index, val_index in tscv.split(X):
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Train model
        start_time = time.time()
        history = model.fit(X_train, y_train, 
                           epochs=50, batch_size=32,
                           validation_data=(X_val, y_val),
                           verbose=0)
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        scores.append(mse)
        training_times.append(training_time)
        
        # Memory tracking (approximate)
        memory_usage.append(model.count_params() * 4 / (1024 ** 2))  # MB
        
    return {
        'avg_mse': np.mean(scores),
        'avg_training_time': np.mean(training_times),
        'model_memory_mb': np.mean(memory_usage)
    }

# %% [3] Backtesting System
class Backtester:
    def __init__(self, model, scaler, initial_balance=10000):
        self.model = model
        self.scaler = scaler
        self.balance = initial_balance
        self.positions = []
        self.portfolio_values = []
        
    def trading_strategy(self, current_price, predicted_price):
        """
        Simple mean-reversion strategy
        """
        if predicted_price > current_price * 1.02:  # 2% expected increase
            return 'buy'
        elif predicted_price < current_price * 0.98:  # 2% expected decrease
            return 'sell'
        else:
            return 'hold'
    
    def run_backtest(self, X_test, y_test, prices):
        """
        Run backtesting simulation
        """
        predictions = self.model.predict(X_test)
        
        # Inverse scaling
        dummy_array = np.zeros((len(predictions), 5))
        dummy_array[:, 0] = predictions.flatten()
        predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        actual_prices = prices[-len(predictions):]
        
        for i in range(len(predictions)):
            action = self.trading_strategy(actual_prices[i], predictions[i])
            
            # Execute trade (simplified)
            if action == 'buy' and self.balance > actual_prices[i]:
                self.positions.append(actual_prices[i])
                self.balance -= actual_prices[i]
            elif action == 'sell' and len(self.positions) > 0:
                bought_price = self.positions.pop(0)
                self.balance += actual_prices[i] - bought_price
                
            # Track portfolio value
            portfolio_value = self.balance + sum(self.positions)
            self.portfolio_values.append(portfolio_value)
        
        return {
            'final_balance': self.balance,
            'portfolio_values': self.portfolio_values,
            'sharpe_ratio': self.calculate_sharpe(),
            'max_drawdown': self.calculate_max_drawdown()
        }
    
    def calculate_sharpe(self, risk_free_rate=0.0):
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        return (np.mean(returns) - risk_free_rate) / np.std(returns)
    
    def calculate_max_drawdown(self):
        peak = self.portfolio_values[0]
        max_dd = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

# %% [4] Model Efficiency Metrics
def model_efficiency_report(y_true, y_pred, training_time, memory_usage):
    """
    Generate comprehensive performance report
    """
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Directional_Accuracy': np.mean(np.sign(y_true[1:] - y_true[:-1]) == 
                                      np.sign(y_pred[1:] - y_pred[:-1])),
        'Training_Time_sec': training_time,
        'Memory_Usage_MB': memory_usage
    }

# %% [5] Deployment Setup
def save_deployment_artifacts(model, scaler):
    model.save('lstm_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    
def load_deployment_artifacts():
    model = tf.keras.models.load_model('lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

app = Flask(__name__)
model, scaler = load_deployment_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sequence = np.array(data['sequence'])
    scaled_sequence = scaler.transform(sequence)
    prediction = model.predict(np.array([scaled_sequence]))
    
    # Inverse scaling
    dummy_array = np.zeros((1, 5))
    dummy_array[:, 0] = prediction.flatten()
    predicted_price = scaler.inverse_transform(dummy_array)[0][0]
    
    return jsonify({
        'prediction': float(predicted_price),
        'timestamp': str(date.today())
    })

# %% [6] Updated Main Execution
if __name__ == "__main__":
    # ... [Previous data loading and processing steps remain the same] ...
    
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
    
    # Create sequences
    time_steps = 7
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Initialize model
    model = create_lstm_model((X.shape[1], X.shape[2]))
    
    # Cross-validated training
    cv_report = train_with_crossval(X, y, model)
    print("\nCross-Validation Report:")
    print(pd.Series(cv_report))
    
    # Final training on full dataset
    history = model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    # Backtesting
    backtester = Backtester(model, scaler)
    backtest_report = backtester.run_backtest(X, y, features['Close'].values)
    
    # Generate predictions for efficiency report
    predictions = model.predict(X)
    
    # Inverse scaling for metrics
    dummy_array = np.zeros((len(predictions), 5))
    dummy_array[:, 0] = predictions.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, 0]
    
    actuals = scaler.inverse_transform(scaled_data)[time_steps:, 0]
    
    # Efficiency report
    efficiency_report = model_efficiency_report(
        actuals, predictions,
        cv_report['avg_training_time'],
        cv_report['model_memory_mb']
    )
    
    print("\nModel Efficiency Report:")
    print(pd.Series(efficiency_report))
    
    # Save deployment artifacts
    save_deployment_artifacts(model, scaler)
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'{ticker} Price Prediction with Backtesting')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Plot portfolio performance
    plt.figure(figsize=(12,6))
    plt.plot(backtest_report['portfolio_values'])
    plt.title('Backtesting Portfolio Performance')
    plt.xlabel('Trades')
    plt.ylabel('Portfolio Value ($)')
    plt.show()

# %% [7] To Run Flask Server
# Execute in terminal: flask run --port=5000
