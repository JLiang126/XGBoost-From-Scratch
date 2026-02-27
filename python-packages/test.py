import my_xgboost # type: ignore
import pandas as pd
import numpy as np

def test() -> int:
    loss = my_xgboost.MSELoss()
    model = my_xgboost.XGBoost(0, 0.0, 0, 0.0, 0.0, 0.0, loss)
    
    model.load_model("sp500_model.txt")
    print("Model loaded")

    data = pd.read_csv("sp500_test.csv")
    features = data.iloc[:, :-1].values 
    returns = data.iloc[:, -1].values 

    preds = []
    for row in features:
        pred = model.predict(list(row))
        preds.append(pred)

    pred_direction = np.sign(preds)
    actual_direction = np.sign(returns)
    accuracy = np.mean(pred_direction == actual_direction)

    print(f"Directional Accuracy: {accuracy * 100:.2f}%")
    return 0

test()