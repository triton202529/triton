# scripts/run_all.py

import os

print("\n🔄 Step 1: Fetching & Preparing Data...")
os.system("python scripts/fetch_and_prepare.py")

print("\n🧠 Step 2: Training AI Models...")
os.system("python scripts/train_model.py")

print("\n📡 Step 3: Generating AI Signals...")
os.system("python services/generate_signals.py")

print("\n🚀 Step 4: Executing Signals (Institutional Chain)...")
print("   Use: python -m services.run_scheduled_paper_cycle")
print("   Legacy auto_execute_signals.py is DEPRECATED (THP-H003).")

print("\n✅ All steps completed. You can now launch the TRITON Control Center:")
print("   streamlit run scripts/control_center.py")
