# How to View Accuracy/Epochs in TensorBoard

## Quick Start

### 1. Start TensorBoard
```bash
tensorboard --logdir=lightning_logs
```

### 2. Open Browser
Go to: **http://localhost:6006**

### 3. View Metrics
Click on the **SCALARS** tab (should be selected by default)

---

## What Metrics Are Available

After the code changes, you'll see these metrics in TensorBoard:

### Training Metrics (per epoch):
- **`train_loss`** - Training loss over epochs
- **`train_accuracy`** - Training accuracy over epochs (NEW!)

### Test/Validation Metrics (per epoch):
- **`test_loss`** - Test loss over epochs
- **`test_accuracy`** - Test accuracy over epochs (same as test_metric)
- **`test_metric`** - Test accuracy (kept for backward compatibility)

---

## TensorBoard Interface Guide

### SCALARS Tab Layout

```
┌─────────────────────────────────────────────────────────────┐
│  TensorBoard - SCALARS                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 train_accuracy                                          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                    ╱─────────          │ │
│  │                          ╱────────╱                    │ │
│  │                 ╱───────╱                              │ │
│  │        ╱───────╱                                       │ │
│  │  ─────╱                                                │ │
│  └───────────────────────────────────────────────────────┘ │
│       Epoch: 0  1  2  3  4  5  6  7  8  9  10             │
│                                                             │
│  📊 test_accuracy                                           │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                                    ╱─────────          │ │
│  │                          ╱────────╱                    │ │
│  │                 ╱───────╱                              │ │
│  │        ╱───────╱                                       │ │
│  │  ─────╱                                                │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  📊 train_loss                                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  ─────╲                                                │ │
│  │        ╲───────╲                                       │ │
│  │                 ╲───────╲                              │ │
│  │                          ╲────────╲                    │ │
│  │                                    ╲─────────          │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Hover over lines** - See exact values at each epoch
2. **Zoom** - Click and drag to zoom into specific epoch ranges
3. **Smoothing slider** - Smooth noisy curves (bottom left)
4. **Download data** - Export as CSV or JSON (top right)

---

## Viewing Multi-Node Training

### Problem: Multiple Runs

With 30 nodes, you'll see 30+ overlapping lines. Here's how to manage:

### Option 1: Filter by Node

In the left sidebar, use the **filter box**:

```
# Show only specific node
127.0.0.1:6666

# Show only accuracy metrics
accuracy

# Show only training metrics
train_
```

### Option 2: Select Specific Runs

In the left sidebar:
1. **Uncheck "Show all runs"**
2. **Manually select** 2-3 nodes to compare
3. Each node will have a different color

### Option 3: View Aggregated Metrics

Create a script to aggregate metrics across all nodes:

```python
# aggregate_metrics.py
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np

def aggregate_tensorboard_logs(log_dir="lightning_logs", output_dir="aggregated_logs"):
    """Aggregate metrics from all nodes."""
    writer = SummaryWriter(output_dir)
    
    all_train_acc = []
    all_test_acc = []
    
    # Read all version directories
    for version_dir in sorted(os.listdir(log_dir)):
        if not version_dir.startswith("version_"):
            continue
        
        version_path = os.path.join(log_dir, version_dir)
        
        # Load events
        ea = event_accumulator.EventAccumulator(version_path)
        ea.Reload()
        
        # Get train accuracy
        if 'train_accuracy' in ea.Tags()['scalars']:
            train_acc = [(s.step, s.value) for s in ea.Scalars('train_accuracy')]
            all_train_acc.append(train_acc)
        
        # Get test accuracy
        if 'test_accuracy' in ea.Tags()['scalars']:
            test_acc = [(s.step, s.value) for s in ea.Scalars('test_accuracy')]
            all_test_acc.append(test_acc)
    
    # Compute average across nodes per epoch
    if all_train_acc:
        max_epochs = max(len(acc) for acc in all_train_acc)
        for epoch in range(max_epochs):
            values = [acc[epoch][1] for acc in all_train_acc if epoch < len(acc)]
            avg_acc = np.mean(values)
            std_acc = np.std(values)
            
            writer.add_scalar('aggregated/train_accuracy_mean', avg_acc, epoch)
            writer.add_scalar('aggregated/train_accuracy_std', std_acc, epoch)
    
    if all_test_acc:
        max_epochs = max(len(acc) for acc in all_test_acc)
        for epoch in range(max_epochs):
            values = [acc[epoch][1] for acc in all_test_acc if epoch < len(acc)]
            avg_acc = np.mean(values)
            std_acc = np.std(values)
            
            writer.add_scalar('aggregated/test_accuracy_mean', avg_acc, epoch)
            writer.add_scalar('aggregated/test_accuracy_std', std_acc, epoch)
    
    writer.close()
    print(f"Aggregated logs saved to {output_dir}")

if __name__ == "__main__":
    aggregate_tensorboard_logs()
```

Run it:
```bash
python aggregate_metrics.py
tensorboard --logdir=aggregated_logs
```

---

## Understanding the Metrics

### Training Accuracy vs Test Accuracy

**Training Accuracy** (`train_accuracy`):
- Measured on the training data
- Usually higher than test accuracy
- Shows how well the model fits the training data

**Test Accuracy** (`test_accuracy`):
- Measured on held-out test data
- More important metric (shows generalization)
- Should increase over epochs

### What Good Training Looks Like

```
Epoch  | Train Acc | Test Acc | Notes
-------|-----------|----------|------------------
0      | 15%       | 12%      | Random guessing (10 classes)
1      | 25%       | 20%      | Learning starts
5      | 60%       | 55%      | Good progress
10     | 80%       | 70%      | Overfitting starts
20     | 95%       | 72%      | Overfitting (train >> test)
```

### Red Flags

⚠️ **Test accuracy decreasing** - Model is overfitting
⚠️ **Train accuracy stuck** - Learning rate too low or model too simple
⚠️ **Both accuracies stuck at ~10%** - Model not learning (CIFAR10 has 10 classes)

---

## Comparing Across Rounds (Federated Learning)

In federated learning, you care about:
1. **Accuracy improvement per round** (not just per epoch)
2. **Convergence across nodes** (all nodes reaching similar accuracy)

### Track Round-Level Metrics

Modify your training to log per-round metrics:

```python
# In your training script
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'logs/rounds/node_{node.addr.replace(":", "_")}')

for round_num in range(total_rounds):
    # Train
    learner.fit()
    
    # Evaluate
    results = learner.evaluate()
    
    # Log to TensorBoard
    writer.add_scalar('Round/test_accuracy', results['test_accuracy'], round_num)
    writer.add_scalar('Round/test_loss', results['test_loss'], round_num)

writer.close()
```

Then view:
```bash
tensorboard --logdir=logs/rounds
```

---

## Advanced: Custom Dashboards

### Create a Custom Metric

```python
# In ResNetCIFAR10.test_step()
self.log("metrics/accuracy_per_class", per_class_acc)
self.log("metrics/confusion_matrix", confusion_matrix)
self.log("metrics/f1_score", f1_score)
```

### Log Histograms

```python
# In ResNetCIFAR10.training_step()
# Log weight distributions
for name, param in self.named_parameters():
    self.logger.experiment.add_histogram(f'weights/{name}', param, self.current_epoch)
```

### Log Images

```python
# In ResNetCIFAR10.test_step()
if batch_id == 0:  # Only first batch
    # Log sample predictions
    grid = torchvision.utils.make_grid(x[:8])
    self.logger.experiment.add_image('test/samples', grid, self.current_epoch)
```

---

## Quick Reference

### Start TensorBoard
```bash
# All logs
tensorboard --logdir=lightning_logs

# Specific run
tensorboard --logdir=lightning_logs/version_0

# Custom port
tensorboard --logdir=lightning_logs --port=6007

# Multiple log directories
tensorboard --logdir_spec=train:logs/train,test:logs/test
```

### Export Data
```bash
# Export to CSV
tensorboard --logdir=lightning_logs --export_to_csv=metrics.csv
```

### Clean Old Logs
```bash
# Keep last 10 versions
find lightning_logs -maxdepth 1 -name "version_*" | sort -V | head -n -10 | xargs rm -rf
```

---

## Summary

**What you'll see in TensorBoard:**

✅ **`train_accuracy`** - Training accuracy per epoch (NEW!)
✅ **`test_accuracy`** - Test accuracy per epoch (same as test_metric)
✅ **`train_loss`** - Training loss per epoch
✅ **`test_loss`** - Test loss per epoch

**How to view:**
1. Run: `tensorboard --logdir=lightning_logs`
2. Open: http://localhost:6006
3. Click: **SCALARS** tab
4. See: Accuracy/loss curves over epochs

**For multi-node training:**
- Use filters to select specific nodes
- Create aggregation script for average metrics
- Consider logging per-round instead of per-epoch

**Expected behavior:**
- Accuracy should increase over epochs
- Loss should decrease over epochs
- Test accuracy should be lower than train accuracy (normal)
