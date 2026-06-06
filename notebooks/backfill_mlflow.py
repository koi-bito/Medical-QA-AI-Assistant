import json
import mlflow
import os

state_path = "models/phi3-medical-lora/checkpoint-2250/trainer_state.json"

if not os.path.exists(state_path):
    print("Error: trainer_state.json not found!")
    exit(1)

with open(state_path, "r") as f:
    state = json.load(f)

log_history = state.get("log_history", [])

# Connect to the MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("medqa-experiments")

print(f"Found {len(log_history)} log entries. Backfilling MLflow...")

# Find the run we created earlier
experiment = mlflow.get_experiment_by_name("medqa-experiments")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string="tags.mlflow.runName = 'phi3-medical-qlora'")

if len(runs) == 0:
    print("Run not found, creating a new one...")
    run_id = None
else:
    run_id = runs.iloc[0].run_id

with mlflow.start_run(run_id=run_id, run_name="phi3-medical-qlora"):
    for entry in log_history:
        step = entry.get("step", 0)
        
        # Training metrics
        if "loss" in entry:
            mlflow.log_metric("train_loss", entry["loss"], step=step)
        if "learning_rate" in entry:
            mlflow.log_metric("learning_rate", entry["learning_rate"], step=step)
            
        # Evaluation metrics
        if "eval_loss" in entry:
            mlflow.log_metric("eval_loss", entry["eval_loss"], step=step)

print("✓ Backfill complete! Please refresh the MLflow UI.")
