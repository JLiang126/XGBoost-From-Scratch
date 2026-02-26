import my_xgboost # type: ignore
import numpy as np

print("--- XGBoost-From-Scratch Verification ---")

try:
    # 1. Initialize the Loss function
    loss = my_xgboost.MSELoss()
    print("MSELoss initialized successfully.")

    # 2. Initialize the Booster (C++ XGBoost Class)
    # Params: num_trees, lr, max_depth, lambda_reg, gamma, min_cover, objective
    model = my_xgboost.XGBoost(
        num_trees=10, 
        learning_rate=0.3, 
        max_depth=5, 
        lambda_reg=1.0, 
        gamma=0.0, 
        min_cover=1.0, 
        objective=loss
    )
    print("XGBoost Booster initialized successfully.")

    # 3. Test a dummy prediction (passing a list to C++ vector)
    dummy_features = [1.5, 2.3, 0.5, 4.2]
    prediction = model.predict(dummy_features)
    print(f"Predict call successful. Raw Output: {prediction}")

    print("\nThe C++ to Python bridge is functional")

except Exception as e:
    print(f"\n ERROR: {e}")