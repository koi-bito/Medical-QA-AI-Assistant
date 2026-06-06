"""
MLflow Setup Test — Day 11
Run this to verify MLflow is working, then open http://localhost:5000 to view results.

Usage:
    python notebooks/test_mlflow.py
    mlflow ui    ← run this in a separate terminal
"""

import mlflow

print("Testing MLflow setup...")

mlflow.set_experiment("medqa-experiments")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 3)
    mlflow.log_param("model", "phi-3-mini")
    mlflow.log_metric("test_accuracy", 0.85)
    mlflow.log_metric("loss", 0.42)

print("✓ Run logged successfully!")
print()
print("To view it, run in a separate terminal:")
print("  mlflow ui")
print("Then open: http://localhost:5000")
