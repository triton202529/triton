# main.py

import os
import subprocess

steps = [
    ("📥 Preprocessing data", "services/preprocess_data.py"),
    ("🧠 Training model(s)", "services/train_model.py"),
    ("📡 Generating signals", "services/generate_signals.py"),
    ("📊 Running portfolio simulation", "services/portfolio_manager.py"),
]

for label, script in steps:
    print(f"\n{label}...")
    result = subprocess.run(["python", script])
    if result.returncode != 0:
        print(f"❌ Error running {script}")
        break

# Final step: launch dashboard
print("\n🚀 Launching Triton Dashboard...")
subprocess.run(["streamlit", "run", "view_results.py"])
