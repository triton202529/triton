#!/usr/bin/env python3
"""
Triton Stress Test Runner
========================

Simple script to run Triton stress tests with different configurations.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_stress_test(scenario="all", quick=False, config="config/stress_test.json"):
    """Run stress tests with specified scenario."""
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    cmd = [sys.executable, "services/stress_test.py", "--config", config]

    if scenario and scenario != "all":
        cmd.extend(["--scenarios", scenario])

    if quick:
        cmd.append("--quick")

    print(f"🧪 Running Triton Stress Test: {scenario}{' (quick)' if quick else ''}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Stress test completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Stress test failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error running stress test: {e}")
        return False


def main():
    """Main function with interactive menu."""
    print("🧪 Triton Stress Test Runner")
    print("=" * 40)
    print("1. Run all stress tests")
    print("2. Run market crash tests only")
    print("3. Run high volatility tests only")
    print("4. Run model failure tests only")
    print("5. Run data corruption tests only")
    print("6. Run system overload tests only")
    print("7. Run quick tests")
    print("8. Exit")

    while True:
        try:
            choice = input("\nSelect option (1-8): ").strip()

            if choice == "1":
                run_stress_test("all")
            elif choice == "2":
                run_stress_test("market_crash")
            elif choice == "3":
                run_stress_test("high_volatility")
            elif choice == "4":
                run_stress_test("model_failure")
            elif choice == "5":
                run_stress_test("data_corruption")
            elif choice == "6":
                run_stress_test("system_overload")
            elif choice == "7":
                run_stress_test("all", quick=True)
            elif choice == "8":
                print("Goodbye! 👋")
                break
            else:
                print("Invalid option. Please select 1-8.")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
