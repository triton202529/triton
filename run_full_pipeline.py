# run_full_pipeline.py
import os
import subprocess

steps = [
    ("🔄 Running preprocessing", "python run_preprocessing.py"),
    ("🧠 Training models", "python train_model.py"),
    ("🔮 Generating predictions", "python generate_predictions.py"),
    ("📡 Generating signals", "python generate_signals.py"),
    ("💼 Simulating portfolio", "python simulate_portfolio.py"),
    ("📊 Launching dashboard", "streamlit run backtest_dashboard.py"),
]

for label, cmd in steps:
    print(f"\n{label}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {cmd}")
        break
