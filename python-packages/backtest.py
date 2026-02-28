import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest(model, test_features, actual_returns):
    # 1. Generate predictions from your C++ engine
    preds = [model.predict(list(row)) for row in test_features]
    
    # 2. Convert to signals (-1 or 1)
    # We use a threshold of 0, but you could use a small buffer
    signals = np.sign(preds)
    
    # 3. Calculate Strategy Returns
    # Note: actual_returns must be the return for the period the 
    # prediction was made for (Day T+1)
    strategy_returns = signals * actual_returns
    
    # 4. Calculate Cumulative Wealth (Starting with $1)
    cumulative_strategy = (1 + strategy_returns).cumprod()
    cumulative_market = (1 + actual_returns).cumprod()
    
    # 5. Calculate Metrics
    win_rate = np.mean(strategy_returns > 0)
    total_return = (cumulative_strategy.iloc[-1] - 1) * 100
    annual_vol = np.std(strategy_returns) * np.sqrt(252)
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)

    print(f"--- Backtest Results ---")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # 6. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_strategy, label='XGBoost Strategy')
    plt.plot(cumulative_market, label='S&P 500 Buy & Hold', alpha=0.6)
    plt.title('Equity Curve: C++ XGBoost vs Market')
    plt.legend()
    plt.grid(True)
    plt.show()
