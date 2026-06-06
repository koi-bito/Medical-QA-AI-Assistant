import json
import matplotlib.pyplot as plt
import os

state_path = "models/phi3-medical-lora/checkpoint-2250/trainer_state.json"
output_path = r"C:\Users\lenovo\.gemini\antigravity-ide\brain\3e0ababa-0da1-4eb5-a743-c8a25c447653\loss_curve.png"

with open(state_path, "r") as f:
    state = json.load(f)

log_history = state.get("log_history", [])

train_steps = []
train_loss = []
eval_steps = []
eval_loss = []

for entry in log_history:
    step = entry.get("step")
    if "loss" in entry:
        train_steps.append(step)
        train_loss.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(step)
        eval_loss.append(entry["eval_loss"])

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label='Training Loss', color='blue', alpha=0.7)
if eval_steps:
    plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', marker='o')

plt.title('Training and Validation Loss over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {output_path}")
