import pandas as pd
import os

def view_predictions():
    file_path = os.path.join("predictions", "latest_predictions.csv")

    if not os.path.exists(file_path):
        print("❌ No predictions found. Run the model training first.")
        return

    df = pd.read_csv(file_path)

    print("\n📈 Latest Model Predictions:\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    view_predictions()
