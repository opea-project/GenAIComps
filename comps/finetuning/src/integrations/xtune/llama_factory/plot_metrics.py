import osAdd commentMore actions
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configuration - Update these paths as needed
#add args
import argparse
parser = argparse.ArgumentParser(description="Plot evaluation metrics")
parser.add_argument("--model_name", type=str, default="Qwen2-VL-2B-Instruct", help="Model name")
parser.add_argument("--experiment_name", type=str, default="finetune_onlyplot_evalloss", help="Experiment name")
args = parser.parse_args()
model_name = args.model_name
experiment_name = args.experiment_name
experiment_name = "finetune_onlyplot_evalloss"
eval_output_dir = f"saves/{model_name}/lora/{experiment_name}/eval"
eval_loss_dir = f"saves/{model_name}/lora/{experiment_name}"

plot_output_dir = os.path.join(eval_output_dir, "plots")
os.makedirs(plot_output_dir, exist_ok=True)

print(f"Looking for evaluation results in: {eval_output_dir}")

# Find all eval directories
eval_dirs = [d for d in os.listdir(eval_output_dir) if d.startswith("eval_checkpoint-")]
print(f"Found {len(eval_dirs)} evaluation directories")

# Extract metrics from all predict_results.json files
metrics_by_checkpoint = {}
checkpoint_numbers = []

for eval_dir in eval_dirs:
    # Extract checkpoint number
    match = re.search(r'checkpoint-(\d+)', eval_dir)
    if not match:
        continue
    
    checkpoint_num = int(match.group(1))
    checkpoint_numbers.append(checkpoint_num)
    
    # Read the predict_results.json file
    results_path = os.path.join(eval_output_dir, eval_dir, "predict_results.json")
    if not os.path.exists(results_path):
        print(f"Warning: No results file found at {results_path}")
        continue
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        metrics_by_checkpoint[checkpoint_num] = results
        print(f"Loaded metrics from checkpoint-{checkpoint_num}")
    except Exception as e:
        print(f"Error reading {results_path}: {e}")

if not metrics_by_checkpoint:
    print("No valid metrics found!")
    exit(1)

# Sort checkpoint numbers
checkpoint_numbers = sorted(checkpoint_numbers)
print(f"Processing checkpoints: {checkpoint_numbers}")

# Organize metrics by name
metric_values = defaultdict(list)

# Collect all unique metric names first
all_metrics = set()
for checkpoint_data in metrics_by_checkpoint.values():
    all_metrics.update(checkpoint_data.keys())

print(f"Found metrics: {', '.join(all_metrics)}")

# Create sorted metric data
for metric in all_metrics:
    for checkpoint_num in checkpoint_numbers:
        if checkpoint_num in metrics_by_checkpoint and metric in metrics_by_checkpoint[checkpoint_num]:
            value = metrics_by_checkpoint[checkpoint_num][metric]
            # Handle both numeric and string values
            if isinstance(value, (int, float)):
                metric_values[metric].append(value)
            else:
                try:
                    metric_values[metric].append(float(value))
                except:
                    # Skip non-numeric metrics
                    print(f"Skipping non-numeric metric: {metric}")
                    break

# Plot each metric individually
for metric, values in metric_values.items():
    if len(values) < 2:  # Skip metrics with insufficient data
        print(f"Skipping {metric} - insufficient data points")
        continue
        
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoint_numbers[:len(values)], values, 'o-', linewidth=2, markersize=8)
    plt.title(f'{metric} across Checkpoints', fontsize=16)
    plt.xlabel('Checkpoint Number', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add data points with values
    for i, (x, y) in enumerate(zip(checkpoint_numbers[:len(values)], values)):
        plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, f'{metric}_plot.png'), dpi=300)
    print(f"Saved plot for {metric}")
    plt.close()

# Create a combined plot with multiple metrics
plt.figure(figsize=(12, 8))
metric_count = 0
legend_entries = []

# Identify the most important metrics to include in the combined plot
priority_metrics = ["BLEU-4", "ROUGE", "METEOR", "accuracy", "f1_score", "rouge1", "rouge2", "rougeL"]
important_metrics = [m for m in priority_metrics if m in metric_values and len(metric_values[m]) >= 2]

# If we don't have priority metrics, just use what we have
if not important_metrics:
    important_metrics = [m for m in metric_values if len(metric_values[m]) >= 2][:5]

# Plot the important metrics
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, metric in enumerate(important_metrics):
    values = metric_values[metric]
    plt.plot(checkpoint_numbers[:len(values)], values, 'o-', 
             linewidth=2, markersize=8, color=colors[i % len(colors)])
    
    # Add values as text annotations
    for j, (x, y) in enumerate(zip(checkpoint_numbers[:len(values)], values)):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=9)
    
    legend_entries.append(metric)
    metric_count += 1
    
    if metric_count >= 5:  # Limit to 5 metrics on one plot to avoid crowding
        break

if metric_count > 0:
    plt.title('Key Metrics across Checkpoints', fontsize=16)
    plt.xlabel('Checkpoint Number', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(legend_entries, fontsize=12, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_dir, 'combined_metrics_plot.png'), dpi=300)
    print(f"Saved combined metrics plot")

# Create a comprehensive CSV with all metrics
csv_path = os.path.join(plot_output_dir, "metrics_summary.csv")
with open(csv_path, 'w') as f:
    # Header row with checkpoint numbers
    f.write("Metric," + ",".join([f"Checkpoint-{num}" for num in checkpoint_numbers]) + "\n")
    
    # Data rows
    for metric in sorted(metric_values.keys()):
        values = metric_values[metric]
        row_data = [metric]
        
        # Add values for each checkpoint
        for i, checkpoint in enumerate(checkpoint_numbers):
            if i < len(values):
                row_data.append(f"{values[i]:.6f}")
            else:
                row_data.append("N/A")
        
        f.write(",".join(row_data) + "\n")

print(f"Generated metrics summary CSV at {csv_path}")

# Now read and plot eval_loss from trainer_state.json
print(f"Reading trainer_state.json from {eval_loss_dir}")
trainer_state_path = os.path.join(eval_loss_dir, "trainer_state.json")

if os.path.exists(trainer_state_path):
    try:
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        # Extract eval_loss and steps from log_history
        eval_steps = []
        eval_losses = []
        
        for entry in trainer_state["log_history"]:
            if "eval_loss" in entry and "step" in entry:
                eval_steps.append(entry["step"])
                eval_losses.append(entry["eval_loss"])
        
        if len(eval_steps) > 1:  # We need at least 2 points to plot a meaningful line
            plt.figure(figsize=(10, 6))
            plt.plot(eval_steps, eval_losses, 'o-', linewidth=2, markersize=8, color='#d62728')
            plt.title('Evaluation Loss During Training', fontsize=16)
            plt.xlabel('Training Step', fontsize=14)
            plt.ylabel('Evaluation Loss', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add data point annotations
            for i, (x, y) in enumerate(zip(eval_steps, eval_losses)):
                plt.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plot_output_dir, 'eval_loss_plot.png'), dpi=300)
            print(f"Saved evaluation loss plot")
            
            # Also save the data to CSV
            csv_path = os.path.join(plot_output_dir, "eval_loss.csv")
            with open(csv_path, 'w') as f:
                f.write("Step,Eval_Loss\n")
                for step, loss in zip(eval_steps, eval_losses):
                    f.write(f"{step},{loss:.6f}\n")
            print(f"Saved evaluation loss data to {csv_path}")
            
        else:
            print("Not enough evaluation loss data points to create a plot")
            
    except Exception as e:
        print(f"Error processing trainer_state.json: {e}")
else:
    print(f"trainer_state.json not found at {trainer_state_path}")

print(f"All plots created in {plot_output_dir}")