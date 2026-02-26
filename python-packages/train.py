import my_xgboost # type: ignore
import pandas as pd
import numpy as np
import time

def train() -> int:

    data = my_xgboost.DataMatrix("sp500_train.csv")
    print("Data Loaded")
    print(f"Engine Data Check:")

    row = data.get_row(0)
    column = data.get_column(0)
    target = data.get_labels()

    print(f" - Rows in C++: {data.get_num_rows()} First Row: {row}")
    print(f" - Features in C++: {data.get_num_columns()} First Column {column[:3]}")
    print(f" - First 3 Labels: {target[:3]}")

    loss = my_xgboost.MSELoss()
    model = my_xgboost.XGBoost(50, 0.1, 3, 1.5, 0.1, 1.0, loss)
    print("Boost Initialised")

    go = input("Train? ")
    if go not in ["Y", "Yes", "y", "YES"]: 
        print("Abort Training")
        return 0

    print("Training")
    start_time = time.time()
    model.train(data, target)
    end_time = time.time()

    print(f"Training Complete in {end_time - start_time:.4f} seconds!")
    return 0

train()