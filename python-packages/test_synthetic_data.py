import my_xgboost # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_synthetic_model():
    # 1. Load the Model
    loss = my_xgboost.MSELoss()
    model = my_xgboost.XGBoost(0, 0.0, 0, 0.0, 0.0, 0.0, loss)
    model.load_model("synthetic_model.txt")
    print("Model loaded successfully!")

    # 2. Load the Test Data
    test_data = pd.read_csv("synthetic_test.csv")
    features = test_data.iloc[:, :-1].values 
    actual_y = test_data.iloc[:, -1].values 

    # 3. Generate Predictions
    preds = np.array([model.predict(list(row)) for row in features])

    # 4. Calculate Metrics
    r2 = r2_score(actual_y, preds)
    rmse = np.sqrt(mean_squared_error(actual_y, preds))

    print(f"\n--- Synthetic Data Evaluation ---")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"RMSE:           {rmse:.4f}")

    # 5. Plot Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_y, preds, alpha=0.5, color='blue', label='Predictions')
    
    # Draw the "Perfect Prediction" diagonal line
    min_val = min(np.min(actual_y), np.min(preds))
    max_val = max(np.max(actual_y), np.max(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    plt.xlabel('Actual Target Values (y)')
    plt.ylabel('Model Predictions')
    plt.title(f'XGBoost Engine Validation\nR² Score: {r2:.4f}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_synthetic_model()